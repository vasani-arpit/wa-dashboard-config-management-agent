import { AIChatAgent } from "@cloudflare/ai-chat";
import { routeAgentRequest, callable } from "agents";
import { streamText, convertToCoreMessages, tool } from "ai";
import { createWorkersAI } from "workers-ai-provider";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { z } from "zod";

import { GitHubService } from "./github-utils";
import {
    updateConfigField,
    parseComments,
    parseFieldValueFromCode
} from "./ast-utils";

export interface Env {
    AI: any;
    GITHUB_TOKEN: string;
    GITHUB_OWNER: string;
    GITHUB_REPO: string;
    GITHUB_BRANCH?: string;
    GEMINI_API_KEY?: string; // Optional if using direct Google API
}

interface StagedChanges {
    customerId: string;
    filePath: string;
    key: string;
    oldValue: any;
    newValue: any;
    isNewFile: boolean;
    updatedCode: string;
    sha?: string;
}

interface AgentState {
    pendingChanges: StagedChanges | null;
}

interface CustomerMeta {
    customerId: string;
    customerName: string | null;
    phone: string | null;
    path: string;
}

const ALLOWED_FIELDS: Record<string, string> = {
    saveMedia: "boolean",
    saveMessages: "boolean",
    billingPlan: "string",
    telegramChatId: "string",
    webhooks: "array"
};

export class ConfigurationAgent extends AIChatAgent<Env, AgentState> {
    initialState: AgentState = { pendingChanges: null };
    private customerIndex: Map<string, CustomerMeta> = new Map();

    private getGitHubService(): GitHubService {
        return new GitHubService(
            this.env.GITHUB_TOKEN,
            this.env.GITHUB_OWNER,
            this.env.GITHUB_REPO,
            this.env.GITHUB_BRANCH || "main"
        );
    }

    /**
     * Lazily fetches files and builds an in-memory index of customer comments.
     */
    private async ensureIndexBuilt(): Promise<void> {
        if (this.customerIndex.size > 0) return;

        const github = this.getGitHubService();
        const configs = await github.listConfigs();
        const index = new Map<string, CustomerMeta>();

        // Concurrency throttle of 10 requests to protect GitHub rate limits
        const limit = 10;
        for (let i = 0; i < configs.length; i += limit) {
            const chunk = configs.slice(i, i + limit);
            await Promise.all(
                chunk.map(async (cfg) => {
                    try {
                        const file = await github.getFile(cfg.path);
                        if (file) {
                            const customerId = cfg.name.replace(".js", "");
                            const { customerName, phone } = parseComments(file.content);
                            const meta: CustomerMeta = {
                                customerId,
                                customerName,
                                phone,
                                path: cfg.path,
                            };
                            index.set(customerId, meta);
                            if (customerName) {
                                index.set(customerName.toLowerCase(), meta);
                            }
                            if (phone) {
                                index.set(phone, meta);
                            }
                        }
                    } catch (err) {
                        console.error(`Index build fail for ${cfg.path}:`, err);
                    }
                })
            );
        }
        this.customerIndex = index;
    }

    private findCustomerInternal(query: { id?: string; customerName?: string; phone?: string }): CustomerMeta | null {
        if (query.id && this.customerIndex.has(query.id)) {
            return this.customerIndex.get(query.id)!;
        }
        if (query.customerName && this.customerIndex.has(query.customerName.toLowerCase())) {
            return this.customerIndex.get(query.customerName.toLowerCase())!;
        }
        if (query.phone) {
            const cleanQueryPhone = query.phone.replace(/[\s\-]/g, "");
            for (const value of this.customerIndex.values()) {
                if (value.phone) {
                    const cleanMetaPhone = value.phone.replace(/[\s\-]/g, "");
                    if (cleanMetaPhone === cleanQueryPhone) return value;
                }
            }
        }
        return null;
    }

    /**
     * Main incoming chat message loop
     */
    async onChatMessage(): Promise<Response> {
        // Select model provider based on configured secrets
        let model;
        if (this.env.GEMINI_API_KEY) {
            const google = createGoogleGenerativeAI({ apiKey: this.env.GEMINI_API_KEY });
            model = google("gemini-1.5-flash");
        } else {
            const workersai = createWorkersAI({ binding: this.env.AI });
            model = workersai("@cf/google/gemini-1.5-flash");
        }

        const systemPrompt = `You are a support configuration agent.

You may only create customer configuration override files and update schema values inside these files.
You may NEVER modify application code or files outside 'src/orgs/'.
If a request demands code changes, gracefully reject the execution and explain that developers must perform the change manually.

ALLOWED_FIELDS SCHEMA:
${JSON.stringify(ALLOWED_FIELDS, null, 2)}

Strictly reject edits on unlisted schema fields (prevent schema expansions).

STEPS:
1. Always deterministic-lookup the customer index using 'findCustomer'. 
2. Call 'updateCustomerConfig' or 'createCustomerConfig' to stage change parameters.
3. Your staging operation acts as a DRY RUN. Ask support staff: "Proceed with commit? (Reply with 'Confirm')"
4. Upon explicit user confirmation (e.g. "Confirm" / "Proceed"), run the 'commitChanges' tool with a clear descriptive message in 'support: [message]' structure.`;

        const result = streamText({
            model,
            system: systemPrompt,
            messages: await convertToModelMessages(this.messages),
            tools: {
                findCustomer: tool({
                    description: "Performs deterministic lookup using customer metadata indexes.",
                    parameters: z.object({
                        id: z.string().optional(),
                        customerName: z.string().optional(),
                        phone: z.string().optional(),
                    }),
                    execute: async (query) => {
                        await this.ensureIndexBuilt();
                        const customer = this.findCustomerInternal(query);
                        if (!customer) {
                            return { success: false, error: "Customer matching details was not located." };
                        }
                        return { success: true, customer };
                    }
                }),

                updateCustomerConfig: tool({
                    description: "Parses configuration AST and stages value overrides for staging review.",
                    parameters: z.object({
                        customerId: z.string(),
                        key: z.string(),
                        value: z.any(),
                    }),
                    execute: async ({ customerId, key, value }) => {
                        try {
                            // Schema Validation
                            const expectedType = ALLOWED_FIELDS[key];
                            if (!expectedType) {
                                return { success: false, error: `Field '${key}' does not exist inside allowed schema parameters.` };
                            }
                            const actualType = Array.isArray(value) ? "array" : typeof value;
                            if (actualType !== expectedType) {
                                return { success: false, error: `Invalid configuration format for '${key}': Expected ${expectedType}, received ${actualType}.` };
                            }

                            const github = this.getGitHubService();
                            const path = `src/orgs/${customerId}.js`;
                            const file = await github.getFile(path);

                            let originalCode = "export default {}";
                            let isNewFile = true;
                            let oldValue: any = "(not set)";

                            if (file) {
                                originalCode = file.content;
                                isNewFile = false;
                                try {
                                    oldValue = parseFieldValueFromCode(originalCode, key) ?? "(not set)";
                                } catch {
                                    oldValue = "(not set)";
                                }
                            }

                            const updatedCode = updateConfigField(originalCode, key, value);

                            // Stage changes safely in Durable Object state
                            this.setState({
                                pendingChanges: {
                                    customerId,
                                    filePath: path,
                                    key,
                                    oldValue,
                                    newValue: value,
                                    isNewFile,
                                    updatedCode,
                                    sha: file?.sha,
                                }
                            });

                            return {
                                success: true,
                                dryRun: true,
                                customerId,
                                filePath: path,
                                key,
                                oldValue,
                                newValue: value,
                                isNewFile,
                            };
                        } catch (error: any) {
                            return { success: false, error: error.message };
                        }
                    }
                }),

                createCustomerConfig: tool({
                    description: "Creates and templates overrides configurations for new customer accounts.",
                    parameters: z.object({
                        customerId: z.string(),
                    }),
                    execute: async ({ customerId }) => {
                        try {
                            const github = this.getGitHubService();
                            const path = `src/orgs/${customerId}.js`;
                            const file = await github.getFile(path);

                            if (file) {
                                return { success: false, error: `Override config for ${customerId} already exists.` };
                            }

                            const template = `// file: src/orgs/${customerId}.js\nexport default {}\n`;

                            this.setState({
                                pendingChanges: {
                                    customerId,
                                    filePath: path,
                                    key: "creation",
                                    oldValue: null,
                                    newValue: "export default {}",
                                    isNewFile: true,
                                    updatedCode: template,
                                }
                            });

                            return {
                                success: true,
                                dryRun: true,
                                customerId,
                                filePath: path,
                                isNewFile: true,
                            };
                        } catch (error: any) {
                            return { success: false, error: error.message };
                        }
                    }
                }),

                commitChanges: tool({
                    description: "Commits staged overrides to the main GitHub branch directly.",
                    parameters: z.object({
                        message: z.string().describe("Commit descriptions structural prefix standard: support: <action>"),
                    }),
                    execute: async ({ message }) => {
                        const staged = this.state.pendingChanges;
                        if (!staged) {
                            return { success: false, error: "No changes staged. Perform updates first." };
                        }

                        try {
                            const github = this.getGitHubService();
                            await github.updateFile(
                                staged.filePath,
                                staged.updatedCode,
                                message,
                                staged.sha
                            );

                            // Clear local changes
                            this.setState({ pendingChanges: null });

                            // Rebuild Index dynamically
                            this.customerIndex.clear();

                            return {
                                success: true,
                                message: "Successfully committed customer configurations to branch.",
                                commitMessage: message,
                                filePath: staged.filePath
                            };
                        } catch (error: any) {
                            return { success: false, error: error.message };
                        }
                    }
                })
            }
        });

        return result.toUIMessageStreamResponse();
    }

    /**
     * Programmatic Trigger Endpoint API Execution
     */
    @callable()
    async triggerProgrammatically(prompt: string): Promise<{ success: boolean; result: any }> {
        const run = await this.saveMessages([
            {
                id: crypto.randomUUID(),
                role: "user",
                parts: [{ type: "text", text: prompt }],
            }
        ]);
        return { success: true, result: run };
    }
}

/**
 * Worker Entry Point Router Bindings
 */
export default {
    async fetch(request: Request, env: Env) {
        const response = await routeAgentRequest(request, env);
        if (response) return response;

        // Direct HTTP API trigger integration (POST /api/trigger)
        const url = new URL(request.url);
        if (request.method === "POST" && url.pathname === "/api/trigger") {
            try {
                const { prompt } = (await request.json()) as { prompt: string };
                const id = env.ConfigurationAgent.idFromName("default-harness");
                const stub = env.ConfigurationAgent.get(id);

                const rpcResult = await stub.triggerProgrammatically(prompt);
                return new Response(JSON.stringify(rpcResult), {
                    headers: { "Content-Type": "application/json" }
                });
            } catch (e: any) {
                return new Response(JSON.stringify({ success: false, error: e.message }), {
                    status: 500,
                    headers: { "Content-Type": "application/json" }
                });
            }
        }

        return new Response("Not found", { status: 404 });
    }
};