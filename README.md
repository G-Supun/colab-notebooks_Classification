# SMS Promotional vs Transactional Classifier

A lightweight SMS text classification system using a distilled 1D-CNN student model.

The model classifies SMS messages into two categories:

- promotional
- transactional

The project was developed using a DistilBERT teacher model and knowledge distillation into a lightweight 1D-CNN student model.

## Project Structure

```text
colab-notebooks_Classification/
│
├── classifierPipline.ipynb
│
├── model/
│   ├── student_cnn.pt
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── src/
│   └── classifier.py
│
├── mcp/
│   └── server.py
│
├── requirements.txt
├── README.md
└── .gitignore
