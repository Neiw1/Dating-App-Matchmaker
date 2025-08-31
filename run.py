from cupid.server.main import app

def main():
    app.run("0.0.0.0", 5003)
    print("Hello from cupid-matcher!")


if __name__ == "__main__":
    main()
