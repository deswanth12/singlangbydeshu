import json

import pytest

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"SignLang AI Vision" in rv.data


def test_gestures_api(client):
    rv = client.get('/api/gestures')
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert 'gestures' in data
    assert len(data['gestures']) > 0


def test_stats_api(client):
    rv = client.get('/api/stats')
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert 'uptime_seconds' in data
    assert 'total_classifications' in data


def test_classify_endpoint(client):
    sample_landmarks = [[[0.0, 0.0, 0.0]] * 21]
    rv = client.post('/classify', json={'landmarks': sample_landmarks})
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert 'label' in data
    assert 'total_classified' in data


def test_history_api(client):
    # GET history
    rv_get = client.get('/api/history')
    assert rv_get.status_code == 200
    data_get = json.loads(rv_get.data)
    assert 'history' in data_get

    # DELETE history
    rv_del = client.delete('/api/history')
    assert rv_del.status_code == 200
    data_del = json.loads(rv_del.data)
    assert data_del['status'] == 'cleared'


def test_practice_api(client):
    # POST practice record
    rv_post = client.post('/api/practice', json={
        'target_sign': 'PEACE',
        'matched_sign': 'PEACE',
        'score': 100,
        'streak': 1
    })
    assert rv_post.status_code == 200

    # GET practice history
    rv_get = client.get('/api/practice')
    assert rv_get.status_code == 200
    data_get = json.loads(rv_get.data)
    assert 'practice_history' in data_get
    assert len(data_get['practice_history']) > 0
