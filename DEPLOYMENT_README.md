# 🤖 Urdu Conversational Chatbot - Streamlit Cloud Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🌟 Features

- **🎯 Top-K Response Generation**: Generate 1-5 different response options
- **🧠 AI Attention Visualization**: See how the AI focuses on words
- **📊 Educational Dashboard**: User-friendly analysis for everyone
- **🎨 Beautiful UI**: Enhanced glassmorphism design
- **⚡ Real-time Chat**: Instant Urdu conversation

## 🚀 Live Demo

Visit the live app: [Your App URL Here](https://your-app-name.streamlit.app)

## 📋 Deployment Instructions

### Option 1: One-Click Deploy
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set main file as `app.py`
6. Deploy!

### Option 2: Manual Setup
1. Clone the repository:
```bash
git clone https://github.com/hamxagithub/Urdu-Conversational-Chatbot-.git
cd Urdu-Conversational-Chatbot-
```

2. Install dependencies:
```bash
pip install -r requirements-streamlit.txt
```

3. Run locally:
```bash
streamlit run app.py
```

## 🛠️ Configuration

The app includes optimized settings for Streamlit Cloud in `.streamlit/config.toml`:
- CPU-optimized PyTorch builds
- Memory-efficient configurations  
- Cloud-friendly theme settings

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── files/                 # Model and tokenizer files
│   ├── best_model.pth    # Trained model weights
│   ├── tokenizer.model   # SentencePiece tokenizer
│   └── vocab_mapping.pkl # Vocabulary mapping
├── .streamlit/           # Streamlit configuration
├── requirements-streamlit.txt # Cloud-optimized dependencies
└── README.md            # This file
```

## 🤖 Model Information

- **Architecture**: Transformer-based encoder-decoder
- **Language**: Urdu (اردو)
- **Features**: Attention analysis, Top-K sampling, Educational visualizations
- **Size**: Optimized for cloud deployment

## 🎯 Usage

1. **Enter Urdu text** in the input field
2. **Adjust settings** in the sidebar:
   - Response count (K): 1-5 options
   - Generation method: Nucleus, Beam Search, etc.
   - Analysis features: Attention visualization
3. **View results**: Multiple response options and AI analysis
4. **Learn**: Educational explanations of how AI works

## 🔧 Advanced Features

- **Top-K Responses**: See multiple AI-generated alternatives
- **Attention Maps**: Visual heatmaps of AI focus
- **Processing Flow**: Step-by-step AI thinking visualization  
- **Educational Mode**: Explanations for beginners

## 📊 Performance

- **Response Time**: ~2-3 seconds per query
- **Concurrent Users**: Optimized for multiple users
- **Memory Usage**: Cloud-optimized for Streamlit limits

## 🛡️ Privacy

- No data storage: Conversations are not saved
- Local processing: All AI computation happens in real-time
- Secure: No personal information required

## 📝 License

This project is open source. See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes  
4. Submit a pull request

## 📧 Contact

- GitHub: [@hamxagithub](https://github.com/hamxagithub)
- Repository: [Urdu-Conversational-Chatbot-](https://github.com/hamxagithub/Urdu-Conversational-Chatbot-)

---

**Made with ❤️ for the Urdu-speaking community**