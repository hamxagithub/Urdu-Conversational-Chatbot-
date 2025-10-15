# 🤖 Urdu Conversational Chatbot - اردو چیٹ بوٹ

A sophisticated Urdu conversational chatbot built with Transformer architecture from scratch using PyTorch.

## 🚀 Features

- **Custom Transformer Architecture**: Built from scratch encoder-decoder model
- **Urdu Language Support**: Specialized tokenization and text processing for Urdu
- **Interactive UI**: Streamlit-based web interface with RTL text support
- **Multiple Decoding Strategies**: Greedy and beam search generation
- **Real-time Chat**: Responsive conversation interface

## 📋 Requirements

```bash
torch>=1.9.0
streamlit>=1.25.0
sentencepiece>=0.1.97
numpy>=1.21.0
pandas>=1.3.0
```

## 🛠️ Installation & Setup

### Method 1: Quick Setup (Recommended)

1. **Clone the repository**:
```bash
git clone https://github.com/hamxagithub/Urdu-Conversational-Chatbot-.git
cd Urdu-Conversational-Chatbot-
```

2. **Install dependencies**:
```bash
pip install torch streamlit sentencepiece numpy pandas
```

3. **Download model files** (if using Git LFS):
```bash
git lfs pull
```

4. **Run the chatbot**:
```bash
streamlit run app.py
```

### Method 2: Manual Model Download

If you encounter issues with large files:

1. Download model files manually from [Google Drive/OneDrive link]
2. Place them in the `files/` directory:
   - `best_model.pth` (92MB)
   - `best_model.pkl` (99MB)
   - `tokenizer.model`
   - `tokenizer.vocab`
   - `vocab_mapping.pkl`

## 📁 Project Structure

```
Urdu_ChatBot/
├── app.py                          # Main Streamlit application
├── files/                          # Model and tokenizer files
│   ├── best_model.pth              # Primary model weights (PyTorch)
│   ├── best_model.pkl              # Fallback model weights (Pickle)
│   ├── tokenizer.model             # SentencePiece model
│   ├── tokenizer.vocab             # SentencePiece vocabulary
│   └── vocab_mapping.pkl           # Token-to-ID mapping
├── check_model.py                  # Model validation script
├── check_model_detailed.py         # Detailed model analysis
└── *.ipynb                         # Training notebooks
```

## 🎯 Usage

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Access the web interface**:
   - Open your browser to `http://localhost:8501`
   - Type your message in Urdu
   - Select decoding strategy (Greedy/Beam Search)
   - Enjoy the conversation!

## 🔧 Model Architecture

- **Encoder-Decoder Transformer**: Custom implementation
- **Vocabulary Size**: 8,000 tokens
- **Model Dimensions**: 256 (d_model)
- **Attention Heads**: 2
- **Encoder/Decoder Layers**: 2 each
- **Feed-Forward Dimension**: 1024
- **Max Sequence Length**: 512 tokens

## 📊 Performance

- **Training Dataset**: 20,000 Urdu conversation pairs
- **Vocabulary Coverage**: Comprehensive Urdu text normalization
- **Response Quality**: Natural conversational responses
- **Inference Speed**: ~0.1-0.8 seconds per response

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built for academic research in Urdu NLP
- Uses SentencePiece for efficient tokenization
- Streamlit for interactive web interface

## 📞 Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Made with ❤️ for the Urdu language community**