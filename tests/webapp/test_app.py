import pytest

# Basic GET tests for main pages
def test_home_page(client):
    """Test the homepage."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to Our AI Showcase!" in response.data
    assert b"AI Model Gallery" in response.data # From header or nav

def test_gallery_page(client):
    """Test the model gallery page."""
    response = client.get('/gallery')
    assert response.status_code == 200
    assert b"Explore Our AI Models" in response.data # Header on gallery page
    assert b"Emotion Detector" in response.data # Check if a model name is present

def test_emotion_detector_demo_page_get(client):
    """Test GET request for Emotion Detector demo page."""
    response = client.get('/demo/emotion-detector')
    assert response.status_code == 200
    assert b"Emotion Detector Demo" in response.data
    assert b"Enter text:" in response.data

def test_mobilenet_demo_page_get(client):
    """Test GET request for MobileNet demo page."""
    response = client.get('/demo/mobilenet')
    assert response.status_code == 200
    assert b"MobileNet Image Classification Demo" in response.data
    assert b"Upload an image:" in response.data

def test_tinybert_demo_page_get(client):
    """Test GET request for TinyBERT demo page."""
    response = client.get('/demo/tinybert')
    assert response.status_code == 200
    assert b"TinyBERT Sentiment Analysis Demo" in response.data
    assert b"Enter text for sentiment analysis:" in response.data

# Basic POST test for one demo endpoint
def test_emotion_detector_demo_post(client):
    """Test POST request to Emotion Detector demo."""
    test_text = "I am feeling ecstatic today!"
    response = client.post('/demo/emotion-detector', data={'text_input': test_text})
    assert response.status_code == 200
    # Check if the results section is present (implies successful processing)
    # The exact predicted label can vary, so check for structure or input text reflection.
    assert b"Results:" in response.data
    assert test_text.encode('utf-8') in response.data # Check if input text is shown in output

def test_tinybert_demo_post(client):
    """Test POST request to TinyBERT Sentiment demo."""
    test_text = "This is a wonderful library, I love it!"
    response = client.post('/demo/tinybert', data={'text_input': test_text})
    assert response.status_code == 200
    assert b"Results:" in response.data
    assert test_text.encode('utf-8') in response.data
    # Depending on the (untrained) model, it might predict 'positive' or 'negative'
    # For now, just checking for the results structure is safer.
    # assert b"Predicted Sentiment: positive" in response.data


# Placeholder for MobileNet POST test - requires file upload handling
# which is more complex with test_client and typically needs an actual file.
# @pytest.mark.skip(reason="MobileNet POST test requires file handling setup")
# def test_mobilenet_demo_post(client):
#     # This test would need to simulate a file upload.
#     # Example (simplified, may need more robust handling):
#     # from io import BytesIO
#     # data = {
#     # 'image_file': (BytesIO(b"dummy_image_content"), 'test.jpg')
#     # }
#     # response = client.post('/demo/mobilenet', data=data, content_type='multipart/form-data')
#     # assert response.status_code == 200
#     # assert b"Top 5 Predictions:" in response.data
#     pass
