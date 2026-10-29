import base64

from fastapi.testclient import TestClient

from serving.app import create_app

KEY = "pilot-" + "x" * 32


def payload(visual):
    return {
        "model": "shelfscan-student",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this cover"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(visual.image_bytes).decode(),
                        },
                    },
                ],
            }
        ],
    }


def test_serving_requires_auth_on_health_and_completions(visual):
    client = TestClient(create_app(lambda *args: "{}", KEY))
    assert client.get("/health").status_code == 401
    assert client.post("/v1/chat/completions", json=payload(visual)).status_code == 401
    assert client.get("/health", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_authenticated_completion_uses_model_callback(visual):
    calls = []

    def generate(raw, prompt, tokens):
        calls.append((raw, prompt, tokens))
        return '{"title":"Test"}'

    client = TestClient(create_app(generate, KEY))
    response = client.post(
        "/v1/chat/completions", json=payload(visual), headers={"Authorization": f"Bearer {KEY}"}
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == '{"title":"Test"}'
    assert calls[0][1:] == ("Read this cover", 450)


def test_remote_image_urls_are_rejected_without_fetch(visual):
    request = payload(visual)
    request["messages"][0]["content"][1]["image_url"]["url"] = "http://169.254.169.254/"
    client = TestClient(create_app(lambda *args: "{}", KEY))
    response = client.post(
        "/v1/chat/completions", json=request, headers={"Authorization": f"Bearer {KEY}"}
    )
    assert response.status_code == 400


def test_inference_errors_do_not_expose_credentials_or_provider_messages(visual):
    def broken(*args):
        raise RuntimeError("private access details")

    client = TestClient(create_app(broken, KEY))
    response = client.post(
        "/v1/chat/completions", json=payload(visual), headers={"Authorization": f"Bearer {KEY}"}
    )
    assert response.status_code == 503
    assert "private access details" not in response.text
