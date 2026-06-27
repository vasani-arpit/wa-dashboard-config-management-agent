import { AIChatAgent } from "@cloudflare/ai-chat";
import { routeAgentRequest, callable } from "agents";
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
    GEMINI_API_KEY: string; // Optional if using direct Google API,
    CLOUDFLARE_ACCOUNT_ID: string;
    CLOUDFLARE_TOKEN: string;
    ConfigurationAgent: DurableObjectNamespace;
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

interface GeminiPart {
    text?: string;
    functionCall?: { name: string; args: Record<string, any> };
    functionResponse?: { name: string; response: Record<string, any> };
}

interface GeminiContent {
    role: "user" | "model";
    parts: GeminiPart[];
}

interface GeminiFunctionDeclaration {
    name: string;
    description: string;
    parameters: {
        type: "OBJECT";
        properties: Record<string, any>;
        required?: string[];
    };
}

async function callGemini(
    accountId: string,
    token: string,
    systemInstruction: string,
    contents: GeminiContent[],
    tools: GeminiFunctionDeclaration[]
): Promise<any> {
    const url = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run`;

    const body: any = {
        model: "google/gemini-2.5-flash-lite",
        input: {
            system_instruction: { parts: [{ text: systemInstruction }] },
            contents,
            generationConfig: { temperature: 0.2, maxOutputTokens: 2048 },
            ...(tools.length > 0 && { tools: [{ functionDeclarations: tools }] }),
        },
    };

    const res = await fetch(url, {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
    });

    if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Cloudflare AI error ${res.status}: ${errText}`);
    }

    const json = await res.json() as any;

    // Cloudflare wraps the Gemini response under `result`
    if (!json.success) {
        throw new Error(`Cloudflare AI error: ${JSON.stringify(json.errors)}`);
    }

    return json.result;
}

// Convert AIChatAgent message history → Gemini contents array
function toGeminiContents(messages: any[]): GeminiContent[] {
    const result: GeminiContent[] = [];
    for (const msg of messages) {
        const role = msg.role === "assistant" ? "model" : "user";
        const text = typeof msg.content === "string"
            ? msg.content
            : (msg.parts ?? []).map((p: any) => p.text ?? "").join("");
        if (text.trim()) {
            result.push({ role, parts: [{ text }] });
        }
    }
    return result;
}

// Tool declarations for Gemini function calling
const GEMINI_TOOLS: GeminiFunctionDeclaration[] = [
    {
        name: "findCustomer",
        description: "Performs deterministic lookup using customer metadata indexes.",
        parameters: {
            type: "OBJECT",
            properties: {
                id: { type: "string", description: "The numeric customer/org ID" },
                customerName: { type: "string", description: "Customer name from the file comment" },
                phone: { type: "string", description: "WhatsApp phone number" },
            },
        },
    },
    {
        name: "updateCustomerConfig",
        description: "Parses configuration AST and stages value overrides for staging review.",
        parameters: {
            type: "OBJECT",
            properties: {
                customerId: { type: "string", description: "The org/customer ID (filename without .js)" },
                key: { type: "string", description: `One of: ${Object.keys(ALLOWED_FIELDS).join(", ")}` },
                value: { description: "New value. Must match the field type. Can be null for nullable fields like telegramChatId." },
            },
            required: ["customerId", "key", "value"],
        },
    },
    {
        name: "createCustomerConfig",
        description: "Creates and templates overrides configurations for new customer accounts.",
        parameters: {
            type: "OBJECT",
            properties: {
                customerId: { type: "string", description: "The new customer/org ID" },
            },
            required: ["customerId"],
        },
    },
    {
        name: "commitChanges",
        description: "Commits staged overrides to the main GitHub branch directly.",
        parameters: {
            type: "OBJECT",
            properties: {
                message: { type: "string", description: "Commit message — must start with 'support: '" },
            },
            required: ["message"],
        },
    },
];

export class ConfigurationAgent extends AIChatAgent<Env, AgentState> {
    initialState: AgentState = { pendingChanges: null };
    private customerIndex: Map<string, CustomerMeta> = new Map();

    private getGitHubService(): GitHubService {
        return new GitHubService(
            this.env.GITHUB_TOKEN,
            this.env.GITHUB_OWNER,
            this.env.GITHUB_REPO,
            this.env.GITHUB_BRANCH || "master"
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

    // ── Tool executor — runs the function the model requested ──────────────
    private async executeTool(name: string, args: Record<string, any>): Promise<Record<string, any>> {
        try {
            if (name === "findCustomer") {
                if (args.id === "default") {
                    return { success: true, customer: { customerId: "default", customerName: null, phone: null, path: "src/orgs/default.js" } };
                }
                await this.ensureIndexBuilt();
                const customer = this.findCustomerInternal(args);
                if (!customer) {
                    return { success: false, error: "Customer matching details was not located." };
                }
                return { success: true, customer };
            }

            if (name === "updateCustomerConfig") {
                const { customerId, key, value } = args;
                // Schema Validation
                const expectedType = ALLOWED_FIELDS[key];
                if (!expectedType) {
                    return { success: false, error: `Field '${key}' does not exist inside allowed schema parameters.` };
                }
                const actualType = Array.isArray(value) ? "array" : typeof value;
                const isNullableField = expectedType === "string" && value === null;
                if (!isNullableField && actualType !== expectedType) {
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
                        /* ignore */
                    }
                }

                // Skip if existing values to prevent unnecessary updates
                if (!isNewFile && oldValue === value) {
                    return {
                        success: true,
                        noChange: true,
                        message: `'${key}' for customer ${customerId} is already set to ${JSON.stringify(value)}. No update required.`
                    };
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
            }
            if (name === "createCustomerConfig") {
                const { customerId } = args;
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
                        updatedCode: template
                    }
                });

                return {
                    success: true,
                    dryRun: true,
                    customerId,
                    filePath: path,
                    isNewFile: true,
                };
            }

            if (name === "commitChanges") {
                const staged = this.state.pendingChanges;
                if (!staged) {
                    return { success: false, error: "No changes staged. Perform updates first." };
                }
                const github = this.getGitHubService();
                await github.updateFile(
                    staged.filePath,
                    staged.updatedCode,
                    args.message,
                    staged.sha
                );

                // Clear local changes
                this.setState({ pendingChanges: null });

                // Rebuild Index dynamically
                this.customerIndex.clear();

                return {
                    success: true,
                    message: "Successfully committed customer configurations to branch.",
                    commitMessage: args.message,
                    filePath: staged.filePath
                };
            }

            return { success: false, error: `Unknown tool: ${name}` };
        } catch (err: any) {
            return { success: false, error: err.message };
        }
    }

    private async runAgentLoop(contents: GeminiContent[]): Promise<string> {
        const systemPrompt = `You are a support configuration agent.
IMPORTANT: Never ask clarifying questions. All information needed is in the user's message.
If a customer ID, name, or phone is provided, call findCustomer immediately with that information.
If the user refers to 'default.js' or 'default config' or 'default file', call findCustomer with id: "default".
You may only create customer configuration override files and update schema values inside these files.
You may NEVER modify application code or files outside 'src/orgs/'.
If a request demands code changes, gracefully reject the execution and explain that developers must perform the change manually.
ALLOWED_FIELDS SCHEMA:
${JSON.stringify(ALLOWED_FIELDS, null, 2)}
Strictly reject edits on unlisted schema fields (prevent schema expansions).
STEPS:
1. Always deterministic-lookup the customer index using 'findCustomer'.
2. Call 'updateCustomerConfig' or 'createCustomerConfig' to stage change parameters.
   CRITICAL: Pass the EXACT value from the user's request. If user says "set X to false", pass boolean false. If user says "set X to true", pass boolean true. If user says "set X to null", pass null. Never infer or substitute the value.
   If the tool returns noChange: true, reply with only the message from the tool result and STOP. Do not ask for confirmation.
3. Your staging operation acts as a DRY RUN. Ask support staff: "Proceed with commit? (Reply with 'Confirm')"
4. Upon explicit user confirmation (e.g. "Confirm" / "yes"), run the 'commitChanges' tool with a clear descriptive message in 'support: [message]' structure.`;

        let loopContents = [...contents];
        let finalText = "";
        const MAX_ITERATIONS = 8;

        for (let i = 0; i < MAX_ITERATIONS; i++) {
            const geminiResponse = await callGemini(this.env.CLOUDFLARE_ACCOUNT_ID, this.env.CLOUDFLARE_TOKEN, systemPrompt, loopContents, GEMINI_TOOLS);
            const candidate = geminiResponse?.candidates?.[0];
            const modelParts: GeminiPart[] = candidate?.content?.parts ?? [];

            const funcCall = modelParts.find((p: GeminiPart) => p.functionCall);
            if (!funcCall?.functionCall) {
                finalText = modelParts.map((p: GeminiPart) => p.text ?? "").join("");
                break;
            }

            const { name, args } = funcCall.functionCall;
            console.log(`[agent] tool call: ${name}`, args);
            const toolResult = await this.executeTool(name, args);
            console.log(`[agent] tool result:`, toolResult);

            loopContents = [
                ...loopContents,
                { role: "model", parts: [{ functionCall: { name, args } }] },
                { role: "user", parts: [{ functionResponse: { name, response: toolResult } }] },
            ];
        }

        return finalText || "Done. Let me know if you need anything else.";
    }

    // ── onChatMessage — handles chat UI (browser) requests ─────────────────
    async onChatMessage(): Promise<Response> {
        const contents = toGeminiContents(this.messages);
        const finalText = await this.runAgentLoop(contents);

        // Return in Vercel AI data stream format expected by AIChatAgent UI
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
            start(controller) {
                controller.enqueue(encoder.encode(`0:${JSON.stringify(finalText)}\n`));
                controller.close();
            }
        });

        return new Response(stream, {
            headers: {
                "Content-Type": "text/event-stream",
                "x-vercel-ai-data-stream": "v1",
            }
        });
    }

    // ── triggerProgrammatically — called via /api/trigger or service binding
    // Runs the full agentic loop and returns the text reply directly.
    @callable()
    async triggerProgrammatically(prompt: string): Promise<{ success: boolean; reply: string }> {
        // Short-circuit: if user just confirms a pending change, commit directly
        const normalized = prompt.trim().toLowerCase();
        if (this.state.pendingChanges && ["yes", "confirm", "y"].includes(normalized)) {
            const result = await this.executeTool("commitChanges", {
                message: `support: update ${this.state.pendingChanges.key} for ${this.state.pendingChanges.customerId}`
            });
            return { success: result.success, reply: result.success ? `Changes committed successfully.` : result.error };
        }

        // Normal flow: build history + run agent loop
        const history = toGeminiContents(this.messages ?? []);
        const contents: GeminiContent[] = [
            ...history,
            { role: "user", parts: [{ text: prompt }] }
        ];

        // Run agent loop — calls Gemini + executes tools until text reply
        const reply = await this.runAgentLoop(contents);

        return { success: true, reply };
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

        // POST /api/trigger
        // Body: { "prompt": "set saveMedia to true for customer 123"}
        // Returns: { "success": true, "reply": "..." }
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