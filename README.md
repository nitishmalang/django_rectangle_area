# Rectangle Area Calculator

The Rectangle Area Calculator is a simple Django web application that allows users to calculate the area of a rectangle. It provides a user-friendly interface where users can enter the length and width of the rectangle, and the application will calculate and display the area.

## Features

- Calculate the area of a rectangle based on user input.
- Simple and intuitive user interface.

## Getting Started

To run this application locally, follow these steps:

1. Clone the repository:


git clone https://github.com/your-username/rectangle-area-calculator.git
cd rectangle-area-calculator

2. Create a virtual environment and activate it:


python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required dependencies:


pip install -r requirements.txt


4. Apply the database migrations:



python manage.py migrate


5. Run the development server:



python manage.py runserver



# Open your web browser and navigate to http://127.0.0.1:8000 to access the application.


## How to Use

Enter the length of the rectangle in the provided input field.
Enter the width of the rectangle in the provided input field.
Click the "Calculate" button.
The area of the rectangle will be displayed below the input fields.

