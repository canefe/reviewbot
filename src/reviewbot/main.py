import dotenv

from reviewbot.cli.app import app

dotenv.load_dotenv()

if __name__ == "__main__":
    app()
