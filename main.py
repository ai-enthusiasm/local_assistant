import gradio as gr
from llm import LLM

model_id = "meta-llama/Llama-3.2-1B-Instruct"
assistant = LLM(model_id)

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    with gr.Tab("💬 Trò chuyện với AI"):
        gr.Markdown("### 💡 Hỏi đáp cùng AI")
        with gr.Row():
            with gr.Column(scale=3):
                input_text = gr.Textbox(lines=6, label="✏️ Nhập câu hỏi:", elem_classes=["gradio-textbox"])
                submit_button = gr.Button("🚀 Gửi", elem_classes=["gradio-button"])
            with gr.Column(scale=5):
                output_text = gr.Textbox(label="🎯 AI trả lời:", elem_classes=["gradio-textbox"])
        submit_button.click(fn=assistant.generate_response, inputs=input_text, outputs=output_text)

    with gr.Tab("📄 Tóm tắt tài liệu"):
        gr.Markdown("### 📑 Tóm tắt nhanh tài liệu của bạn")
        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(label="📂 Tải lên tài liệu (chỉ hỗ trợ định dạng .pdf, .docx và .txt)", elem_classes=["gradio-file"], file_types=[".pdf", ".docx", ".txt"])
                analyze_button = gr.Button("📑 Phân tích", elem_classes=["gradio-button"])
            with gr.Column(scale=5):
                analysis_output = gr.Textbox(label="📌 Kết quả tóm tắt:", elem_classes=["gradio-textbox"])
        analyze_button.click(fn=assistant.summarize_file, inputs=file_input, outputs=analysis_output)

# Chạy ứng dụng
demo.queue().launch(share=False, show_api=False, server_port=3001, server_name="0.0.0.0")
