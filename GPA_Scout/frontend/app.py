"""
UTA Course Q&A Agent - Gradio Interface
Run with: python gradio_interface.py
"""

import gradio as gr
import sys
import os
from pathlib import Path

# Import the main agent from your existing script
from main_script import CourseQAAgent, AppConfig, DataConfig, ModelConfig

class GradioCourseAssistant:
    def __init__(self):
        self.agent = None
        self.initialized = False
        
    def initialize_agent(self):
        """Initialize the course agent"""
        try:
            # Create configuration properly
            data_config = DataConfig(
                data_file="project_data.csv",  # Update this path if needed
                index_prefix="uta_production",
                chunk_sizes={'courses': 3, 'professors': 3, 'sections': 3}
            )
            
            config = AppConfig(
                data=data_config,
                log_level="INFO",
                cache_size=1000
            )
            
            self.agent = CourseQAAgent(config)
            self.agent.initialize()
            self.initialized = True
            return "✅ Agent initialized successfully! You can now ask questions about UTA courses."
            
        except Exception as e:
            return f"❌ Failed to initialize agent: {str(e)}"
    
    def process_query(self, query, history):
        """Process user query and return response"""
        if not self.initialized or self.agent is None:
            return "Please initialize the agent first by clicking the 'Initialize Agent' button."
        
        try:
            response = self.agent.process_query(query)
            
            # Format the response for better display
            formatted_response = self._format_response(response)
            return formatted_response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def _format_response(self, response):
        """Format the response for better readability in Gradio"""
        # Replace markdown-style formatting with HTML for better display
        formatted = response.replace("**", "<strong>").replace("**", "</strong>")
        formatted = formatted.replace("\n", "<br>")
        
        # Add some CSS styling
        styled_response = f"""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        '>
            {formatted}
        </div>
        """
        return styled_response

def create_gradio_interface():
    """Create the Gradio interface"""
    
    assistant = GradioCourseAssistant()
    
    with gr.Blocks(
        title="UTA Course Q&A Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chatbot {
            min-height: 500px;
        }
        .example-item {
            cursor: pointer;
            padding: 8px 12px;
            margin: 5px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #f9f9f9;
            transition: all 0.3s ease;
        }
        .example-item:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1 style="color: #667eea; margin-bottom: 10px;">🎓 UTA Course Q&A Agent</h1>
                <p style="font-size: 1.2em; color: #666;">
                    <strong>Enhanced with Course Analytics, Professor Insights & Grade Lookups</strong>
                </p>
                <p style="color: #888;">
                    Ask about courses, professors, grades, comparisons, and more!
                </p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🚀 Initialization")
                init_btn = gr.Button(
                    "Initialize Agent", 
                    variant="primary",
                    size="lg"
                )
                init_status = gr.Textbox(
                    label="Status",
                    placeholder="Click 'Initialize Agent' to start...",
                    interactive=False,
                    show_label=False
                )
                
                gr.Markdown("### 💡 Example Queries")
                examples = gr.Dataset(
                    components=[gr.Textbox(visible=False)],
                    samples=[
                        ["CSE 5334"],
                        ["Compare CSE 5334 and CSE 5330"],
                        ["Courses by John Smith"],
                        ["CSE 5334 grades Spring 2023"],
                        ["Easy computer science courses"],
                        ["Professor John Smith"],
                        ["History of CSE 5334"],
                        ["CSE 5334 with Professor John"]
                    ],
                    label="Click an example to try it!",
                    samples_per_page=8
                )
                
                gr.Markdown("### 📊 Features")
                with gr.Accordion("Click to see features", open=False):
                    gr.Markdown("""
                    - **Course Lookups**: Detailed course information with GPA, pass rates
                    - **Professor Analytics**: Teaching styles and grade distributions  
                    - **Grade Lookups**: Specific term/professor grade distributions
                    - **Course Comparisons**: Side-by-side course comparisons
                    - **Topic Search**: Find courses by topic/difficulty
                    - **Historical Data**: Course offering history over time
                    """)
                
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Course Q&A Chat",
                    height=500,
                    show_copy_button=True,
                    placeholder="Ask me anything about UTA courses...",
                    type="messages"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Type your question about UTA courses here...",
                        scale=4,
                        container=False
                    )
                    submit_btn = gr.Button("Send 📤", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    restart_btn = gr.Button("🔄 Restart Agent", variant="secondary")
        
        # Event handlers
        init_btn.click(
            fn=assistant.initialize_agent,
            outputs=init_status
        )
        
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
                
            response = assistant.process_query(message, chat_history)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            return "", chat_history
        
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            fn=lambda: None,
            inputs=[],
            outputs=[chatbot],
            queue=False
        ).then(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        restart_btn.click(
            fn=lambda: ("Agent restarting...", []),
            outputs=[init_status, chatbot]
        ).then(
            fn=assistant.initialize_agent,
            outputs=init_status
        )
        
        def load_example(example):
            return example[0]
        
        examples.click(
            fn=load_example,
            inputs=[examples],
            outputs=[msg]
        )
    
    return demo, assistant

def main():
    """Main function to launch Gradio app"""
    print("🚀 Starting UTA Course Q&A Agent with Gradio...")
    print("📁 Make sure your project_data.csv is in the same directory")
    
    # Check if data file exists
    data_files_to_check = [
        "project_data.csv",
        "/content/project_data.csv",
        "./project_data.csv"
    ]
    
    found_data = False
    for data_file in data_files_to_check:
        if os.path.exists(data_file):
            print(f"✅ Found data file: {data_file}")
            found_data = True
            break
    
    if not found_data:
        print("⚠️  Warning: Data file not found.")
        print("Please make sure project_data.csv is in one of these locations:")
        for data_file in data_files_to_check:
            print(f"   - {data_file}")
    
    demo, assistant = create_gradio_interface()
    
    # Launch the interface
    print("🌐 Launching Gradio interface...")
    print("📱 Open your browser and go to: http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True for public link
        inbrowser=True,          # Open in browser automatically
        show_error=True
    )

if __name__ == "__main__":
    main()