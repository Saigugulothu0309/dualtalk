# DualTalk v1 - Real-time Sign Language Communication System

DualTalk is a real-time sign language communication system designed to bridge the communication gap between sign language users and non-signers. By utilizing computer vision and deep learning, DualTalk translates hand gestures into text and speech in real-time, while also providing a seamless interface for bidirectional communication.

## 🚀 Features

*   **Real-time Gesture Recognition:** Captures and translates sign language gestures instantly via webcam feed.
*   **Bidirectional Communication:** Facilitates fluid, two-way interaction between users.
*   **Web-based Dashboard:** Responsive and intuitive frontend for easy accessibility.
*   **Configurable Architecture:** Easily customize and add new gestures via YAML configuration files.
*   **Production Ready:** Prepared for rapid deployment using modern cloud platforms.

---

## 🛠️ Project Structure

```text
├── config/
│   └── gestures/          # Configuration files for gesture mapping
├── frontend/              # Frontend assets and UI components
├── models/                # Trained deep learning models and weights
├── src/                   # Core application source code (backend logic, processing)
├── tests/                 # Unit and integration tests
├── web/                   # Web server and routing files
├── .env.example           # Example environment variables file
├── .gitignore             # Git ignore file
├── Procfile               # Heroku deployment configuration
├── app.py                 # Application entry point
├── config.yaml            # Main application configuration file
├── install_requirements.ps1 # PowerShell setup script for Windows
├── railway.json           # Railway deployment configuration
├── requirements.txt       # Python dependencies
├── run.py                 # Script to launch the application
└── runtime.txt            # Python runtime specification
⚙️ Installation & Setup
Prerequisites
Python 3.8+

Webcam (for real-time gesture tracking)

Windows Setup (PowerShell)
You can automatically set up your environment by running the included PowerShell script:

PowerShell
./install_requirements.ps1
Manual Setup
Clone the repository:

Bash
git clone [https://github.com/Saigugulothu0309/dualtalk.git](https://github.com/Saigugulothu0309/dualtalk.git)
cd dualtalk
Create and activate a virtual environment:

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Configure environment variables:

Bash
cp .env.example .env
# Open .env and update your configurations if necessary
💻 Usage
To launch the real-time communication system locally, run:

Bash
python run.py
Open your preferred browser and navigate to the local host address provided in the terminal (typically http://127.0.0.1:5000 or http://localhost:8000).

🌐 Deployment
DualTalk is pre-configured for seamless cloud hosting:

Railway: Configuration is handled automatically via railway.json.

Heroku / Render: Utilizes the provided Procfile and runtime.txt.

🤝 Contributing
Contributions are welcome! If you'd like to add new gestures, optimize the models, or improve the frontend UI:

Fork the repository.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

"TEAM DUALTALK"
