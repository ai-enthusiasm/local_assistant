import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fitz
from docx import Document
import os

class LLM:
    def __init__(self, model_id: str, device="cuda:0"):
        """Khởi tạo mô hình LLM với tokenizer và model từ Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)

    def generate_response(self, user_input, max_new_tokens=4096, temperature=0.7, top_k=50, top_p=0.95):
        """Sinh phản hồi từ mô hình dựa trên tin nhắn đầu vào."""
        messages = [
            {"role": "system", "content": "Bạn là một trợ lý AI thông minh và thân thiện, sẵn sàng hỗ trợ người dùng."},
            {"role": "user", "content": user_input}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        last_assistant_index = generated_text.rfind('assistant\n\n')
        
        if last_assistant_index != -1:
            filtered_content = generated_text[last_assistant_index + len('assistant\n\n'):].strip()
        else:
            filtered_content = "Không tìm thấy nội dung cần thiết."
        
        del inputs
        del outputs
        torch.cuda.empty_cache()

        return filtered_content

    def extract_text_from_pdf(self, file_path):
        """Trích xuất văn bản từ tệp PDF."""
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text

    def summarize_file(self, file, max_size_mb=1, max_new_tokens=4096, temperature=0.7, top_k=50, top_p=0.95):
        """Tóm tắt nội dung từ file văn bản (PDF, DOCX, TXT)."""

        # Kiểm tra kích thước file
        file_size = os.path.getsize(file.name) / (1024 * 1024)
        if file_size > max_size_mb:
            return f"⚠️ Kích thước file quá lớn ({file_size:.2f} MB). Giới hạn là {max_size_mb} MB."

        # Kiểm tra định dạng file
        allowed_extensions = [".pdf", ".docx", ".txt"]
        file_extension = os.path.splitext(file.name)[-1].lower()
        if file_extension not in allowed_extensions:
            return f"⚠️ Định dạng file không hợp lệ ({file_extension}). Vui lòng tải lên file PDF, DOCX hoặc TXT."


        if file.name.endswith(".docx"):
            doc = Document(file.name)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif file.name.endswith(".pdf"):
            content = self.extract_text_from_pdf(file.name)
        else:
            with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        messages = [
            {"role": "system", "content": "Bạn là một trợ lý AI chuyên tóm tắt tài liệu."},
            {"role": "user", "content": f"""
             
            📌 Yêu cầu:

            Tiêu đề nội dung: Đưa ra một tiêu đề chính cho tài liệu.
            Tóm tắt theo phần: Chia tóm tắt thành các mục tương ứng với cấu trúc nội dung.
            Làm nổi bật thông tin quan trọng: Sử dụng bullet points hoặc danh sách đánh số nếu cần.

            🎯 Đầu ra mong muốn:

            Tiêu đề: [Tên tài liệu]

            1. Giới thiệu: [Tóm tắt phần giới thiệu]
            2. Nội dung chính: [Tóm tắt các điểm quan trọng]
            3. Kết luận: [Tóm tắt kết luận]
            Hãy tóm tắt tài liệu sau bằng cách chia nhỏ thành các phần quan trọng:
            
             
            Dưới đây là nội dung của tệp:
            {content}
            """}
        ]
        
        return self.generate_response(messages, max_new_tokens, temperature, top_k, top_p)
