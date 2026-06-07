import { Octokit } from "@octokit/rest";

export class GitHubService {
    private octokit: Octokit;
    private owner: string;
    private repo: string;
    private branch: string;

    constructor(token: string, owner: string, repo: string, branch = "main") {
        this.octokit = new Octokit({ auth: token });
        this.owner = owner;
        this.repo = repo;
        this.branch = branch;
    }

    async listConfigs(): Promise<{ name: string; path: string }[]> {
        try {
            const response = await this.octokit.repos.getContent({
                owner: this.owner,
                repo: this.repo,
                path: "src/orgs",
                ref: this.branch,
            });

            if (!Array.isArray(response.data)) {
                return [];
            }

            return response.data
                .filter(item => item.type === "file" && item.name.endsWith(".js") && item.name !== "default.js")
                .map(item => ({
                    name: item.name,
                    path: item.path,
                }));
        } catch (error: any) {
            if (error.status === 404) return [];
            throw error;
        }
    }

    async getFile(path: string): Promise<{ content: string; sha: string } | null> {
        try {
            const response = await this.octokit.repos.getContent({
                owner: this.owner,
                repo: this.repo,
                path,
                ref: this.branch,
            });

            if (Array.isArray(response.data) || !("content" in response.data)) {
                return null;
            }

            const content = atob(response.data.content.replace(/\s/g, ""));
            return { content, sha: response.data.sha };
        } catch (error: any) {
            if (error.status === 404) return null;
            throw error;
        }
    }

    async updateFile(path: string, content: string, commitMessage: string, sha?: string): Promise<any> {
        // Standard UTF-8 safe base64 encoding
        const encodedContent = btoa(unescape(encodeURIComponent(content)));
        const response = await this.octokit.repos.createOrUpdateFileContents({
            owner: this.owner,
            repo: this.repo,
            path,
            message: commitMessage,
            content: encodedContent,
            sha,
            branch: this.branch,
        });
        return response.data;
    }
}