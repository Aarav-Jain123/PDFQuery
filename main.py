import app
from threading import Thread

if __name__ == "__main__":
    name_of_pdf = str(input("Name of PDF file (must be in same directory): "))
    if not '.pdf' in name_of_pdf:
        name_of_pdf += '.pdf'
    thread = Thread(target=app.create_rag, args=(name_of_pdf,))

    thread.start()

    thread.join()
    