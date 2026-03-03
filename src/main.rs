use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{env, error::Error, process};

#[derive(Debug)]
pub enum ToolError {
    InvalidTool,
    ExecutionFailed,
    InvalidArguments(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::InvalidTool => write!(f, "Invalid tool specified"),
            ToolError::ExecutionFailed => write!(f, "Tool execution failed"),
            ToolError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
        }
    }
}

impl Error for ToolError {}

#[derive(Debug, Deserialize)]
struct Response {
    choices: Vec<Choice>,
}

#[allow(unused)]
#[derive(Debug, Deserialize)]
struct Choice {
    index: usize,
    message: Message,
    finish_reason: String,
}

#[allow(unused)]
#[derive(Debug, Deserialize)]
struct Message {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[allow(unused)]
#[derive(Debug, Deserialize, Serialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    _type: String,
    function: Function,
}

#[derive(Debug, Deserialize, Serialize)]
struct Function {
    name: String,
    #[serde(
        deserialize_with = "deserialize_json_string",
        serialize_with = "serialize_json_string"
    )]
    arguments: Value,
}

pub enum Tools {
    Read(String),
}

impl Tools {
    fn execute(&self) -> Result<String, ToolError> {
        match self {
            Tools::Read(file_path) => {
                std::fs::read_to_string(file_path).map_err(|_| ToolError::ExecutionFailed)
            }
        }
    }
}

fn deserialize_json_string<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    serde_json::from_str(&s).map_err(serde::de::Error::custom)
}

fn serialize_json_string<S>(value: &Value, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let s = serde_json::to_string(value).map_err(serde::ser::Error::custom)?;
    serializer.serialize_str(&s)
}

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let model = env::var("LOCAL_MODEL").unwrap_or("anthropic/claude-haiku-4.5".to_string());

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let mut messages = vec![json!({
        "role": "user",
        "content": args.prompt
    })];

    loop {
        let response: Value = client
            .chat()
            .create_byot(json!({
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read and return the contents of a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path to the file to read"
                                    }
                                },
                                "required": ["file_path"]
                            }
                        }
                    }
                ],
                "messages": messages,
                "model": model,
            }))
            .await?;

        let s_response: Response = serde_json::from_value(response).unwrap();
        if s_response.choices[0].finish_reason == "stop" {
            print!(
                "{}",
                s_response.choices[0]
                    .message
                    .content
                    .clone()
                    .unwrap_or_default()
            );
            break;
        }

        if let Some(tool_calls) = s_response.choices[0].message.tool_calls.as_ref()
            && !tool_calls.is_empty()
        {
            messages.push(json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": s_response.choices[0].message.tool_calls
            }));
            eprintln!("{:?}", tool_calls);
            for tool_call in tool_calls {
                let tool = match tool_call.function.name.as_str() {
                    "Read" => match tool_call.function.arguments["file_path"].as_str() {
                        Some(path) => Tools::Read(path.to_string()),
                        None => {
                            return Err(ToolError::InvalidArguments(
                                tool_call.function.arguments.to_string(),
                            )
                            .into());
                        }
                    },
                    _ => continue,
                };
                let resp = tool.execute()?;
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call.id.clone(),
                    "content": resp,
                }));
                eprintln!("messages: {:?}", messages);
            }
        } else {
            messages.push(json!({
                "role": "assistant",
                "content": s_response.choices[0].message.content.clone().unwrap_or_default(),
            }));
            print!(
                "{}",
                s_response.choices[0]
                    .message
                    .content
                    .clone()
                    .unwrap_or_default()
            );
            break;
        }
    }

    Ok(())
}
