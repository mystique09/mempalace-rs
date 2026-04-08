#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    mempalace_mcp::McpServer::run_stdio().await
}
