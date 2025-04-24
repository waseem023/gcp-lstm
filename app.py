# import os
# from flask import Flask

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Hello TEST TES TE EDSJB!"

# # Use 8081 for local dev, 8080 for Cloud Run
# port = int(os.environ.get("PORT", 8082))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=port)


from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "🚀 Hello from Flask on Cloud Run   WASEEMMMMMEANKND!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)