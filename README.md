# Face Recognition and Verification Web Application

## Overview

This web application is designed for face recognition and verification. It allows users to perform the following actions:

- **Train a Face Model**: Users can train a face recognition model by uploading images of individuals along with their names.

- **Verify a Face**: Users can upload an image and verify if it matches any of the trained faces.

- **Live Streaming**: Users can access live video streaming with face recognition enabled.

## Screenshots

### Home Page

![Screenshot from 2023-09-26 02-48-32](https://github.com/danielAdama/face-rec-web/assets/63624329/903911c9-f9d8-494d-91ec-15ee4899eb04)

### Training Page

![Screenshot from 2023-09-26 02-49-57](https://github.com/danielAdama/face-rec-web/assets/63624329/d4f5df9d-f982-4bca-8749-f7a080e15996)

### Verification Page

![Screenshot from 2023-09-26 02-51-47](https://github.com/danielAdama/face-rec-web/assets/63624329/89c06fbd-abd6-4c9e-8487-a5dcb7fc9a8e)
![Screenshot from 2023-09-26 02-51-52](https://github.com/danielAdama/face-rec-web/assets/63624329/d5861274-b622-41eb-b7f0-6e2980f7eb8e)

## Project Structure

The project is structured as follows:

```
face-rec-web/
│
├── src/
│   ├── app.py              # Main Flask application
│   ├── face_database.py    # Face recognition and database handling
│   ├── templates/          # HTML templates
│   ├── static/             # Static assets (CSS, images, etc.)
│   ├── utils.py/           # Utilities
│   
│
├── docker-compose.yml      # Docker deployment configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/danielAdama/face-recognition-app.git
   cd face-recognition-app
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   Create a `.env` file in the project root and add the following variables:

   ```env
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   ```

4. Start the Flask application:

   ```bash
   python src/app.py
   ```

5. Access the application in your web browser at `http://localhost:8080`.

## Usage

### Training a Face Model

1. Navigate to the home page.

2. Enter the name of the person associated with the uploaded images.

3. Choose multiple images of the person for training (recommended to use more than 4 images).

4. Click the "Train Model" button.

### Verifying a Face

1. Navigate to the verification page.

2. Upload an image for verification (only one image at a time).

3. Click the "Verify Image" button.

### Live Streaming

1. Navigate to the live streaming page.

2. Access the live video stream with face recognition enabled.

## Technologies Used

- Python
- Flask
- OpenCV
- Face Recognition Library
- AWS
- HTML/CSS
- JavaScript (SweetAlert for pop-up notifications)
- Docker

## Deployment

### Docker Compose Deployment

To deploy this application using Docker Compose, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/danielAdama/face-recognition-app.git
   cd face-recognition-app
   ```

2. Ensure you have Docker Compose installed on your system.

3. Run the Docker Compose stack:

   ```bash
   docker-compose up --build
   ```

4. Access the application in your web browser at `http://localhost:8080`.

The web application should now be running in a Docker container.

## Contributors

- [Daniel](https://github.com/danielAdama) [AI & Backend Engineer]
- [Simon](https://github.com/Toviarock1) [Front-end Engineer]

## Additional Information

For more details on this project, refer to the documentation in the [docs](docs/) directory.
```
