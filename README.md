# Flower Classification Web Application

This project is a web application that classifies different types of flowers using a ResNet50 deep learning model. The application consists of a React frontend for image upload and display, and a Flask backend for image processing and classification.

## Prerequisites

- Python 3.10
- Node.js (Latest LTS version recommended)
- Git

## Backend Setup

The backend uses Python 3.10 specifically. If you have multiple Python versions installed, you can check them using:
```bash
py --list
```

### Setting up the Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
   - If you have only Python 3.10 installed:
     ```bash
     python -m venv venv
     ```
   - If you have multiple Python versions:
     ```bash
     py -3.10 -m venv venv
     ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source venv/bin/activate
     ```

4. Install required packages:
```bash
pip install -r requirements.txt
```
If it gives an error during this process, it is OKAY.

5. Running the backend:
   - If you have only Python 3.10:
     ```bash
     python app.py
     ```
   - If you have multiple Python versions:
     ```bash
     py -3.10 app.py
     ```

### Training the Model

To train the model or use existing model:
   - If you have only Python 3.10:
     ```bash
     python model.py train  # Train a new model
     python model.py predict  # Test the model
     ```
   - If you have multiple Python versions:
     ```bash
     py -3.10 model.py train  # Train a new model
     py -3.10 model.py predict  # Test the model
     ```

## Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

## Running the Complete Application

1. Start the backend server (in backend directory):
```bash
py -3.10 app.py
```

2. In a separate terminal, start the frontend (in frontend directory):
```bash
npm run dev
```

3. Access the application at `http://localhost:5173`

## Additional Notes

- The backend server runs on `http://localhost:5000`
- The frontend development server runs on `http://localhost:5173`
- Make sure both servers are running simultaneously for the application to work properly
- The model training requires significant computational resources and may take some time to complete

## Troubleshooting

1. If you encounter Python version issues:
   - Always use `py -3.10` instead of `python` when running Python files
   - Make sure you're using the correct virtual environment

2. If the model isn't working:
   - Ensure you've trained the model using `py -3.10 model.py train`
   - Check that the saved model files exist in the `saved_model` directory

3. If packages are missing:
   - Make sure you're in your virtual environment
   - Run `pip install -r requirements.txt` again