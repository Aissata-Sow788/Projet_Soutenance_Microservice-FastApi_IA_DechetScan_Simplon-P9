from fastapi import FastAPI

app = FastAPI(title="DechetScan IA")


@app.get("/")
def accueil():
    return {
        "message": "Service IA DechetScan opérationnel"
    }
