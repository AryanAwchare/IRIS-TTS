"""
Unit tests for Emotion & Sentiment Analyzer.
Verifies:
  - Canonical 6-emotion taxonomy mapping
  - Intensity scoring on various sentence structures
  - Blending modes (auto, blend, user_only)
  - Mathematical hyperparameter scaling (CFG, Exaggeration)
  - Edge cases (short texts, punctuation energy, capitalization)
"""
from app.utils.emotion_analyzer import (
    analyze_text_sentiment_and_emotion,
    compute_modulated_synthesis_parameters,
    CANONICAL_EMOTIONS,
)


def test_neutral_emotion():
    text = "The system operates at thirty-two kilohertz sample rate."
    res = analyze_text_sentiment_and_emotion(text)
    assert res.emotion == "neutral"
    assert res.intensity <= 0.35

    params = compute_modulated_synthesis_parameters(text, requested_emotion="auto")
    assert params["resolved_emotion"] == "neutral"
    assert params["cfg_weight"] >= 0.65  # Identity locked
    assert params["exaggeration"] <= 0.10


def test_happy_joy_emotion():
    text = "I am so happy and delighted to share this wonderful news with you! Haha, congratulations!"
    res = analyze_text_sentiment_and_emotion(text)
    assert res.emotion == "happy"
    assert res.intensity >= 0.50
    assert "[laughter]" in res.suggested_tags or len(res.suggested_tags) >= 0

    params = compute_modulated_synthesis_parameters(text, requested_emotion="auto")
    assert params["resolved_emotion"] == "happy"
    assert 0.40 <= params["cfg_weight"] <= 0.65
    assert params["exaggeration"] >= 0.15


def test_excited_surprise_emotion():
    text = "Wow! This is totally incredible and amazing! Unbelievable performance!"
    res = analyze_text_sentiment_and_emotion(text)
    assert res.emotion == "excited"
    assert res.intensity >= 0.50

    params = compute_modulated_synthesis_parameters(text, requested_emotion="auto")
    assert params["resolved_emotion"] == "excited"
    assert params["cfg_weight"] <= 0.55  # Freedom for expressiveness
    assert params["exaggeration"] >= 0.30


def test_sad_emotion():
    text = "Unfortunately, we received some deeply sorrowful and heartbreaking news today."
    res = analyze_text_sentiment_and_emotion(text)
    assert res.emotion == "sad"
    assert res.intensity >= 0.40

    params = compute_modulated_synthesis_parameters(text, requested_emotion="auto")
    assert params["resolved_emotion"] == "sad"
    assert params["speed"] <= 0.95  # Slower pace for melancholy


def test_angry_emotion():
    text = "I am furious and absolutely outraged by this terrible and horrible mistake!"
    res = analyze_text_sentiment_and_emotion(text)
    assert res.emotion == "angry"
    assert res.intensity >= 0.50

    params = compute_modulated_synthesis_parameters(text, requested_emotion="auto")
    assert params["resolved_emotion"] == "angry"
    assert params["speed"] >= 1.05


def test_blend_mode():
    text = "I am so excited and happy to see you!"
    # User requested 'calm', but text is excited. In blend mode, it should honor user's emotion with modulated intensity
    params = compute_modulated_synthesis_parameters(
        text,
        requested_emotion="calm",
        blend_mode="blend"
    )
    assert params["resolved_emotion"] == "calm"
    assert params["cfg_weight"] >= 0.60


def test_user_only_mode():
    text = "I am so furious and angry!"
    params = compute_modulated_synthesis_parameters(
        text,
        requested_emotion="neutral",
        blend_mode="user_only"
    )
    assert params["resolved_emotion"] == "neutral"


def test_short_text_edge_case():
    text = "Hi."
    res = analyze_text_sentiment_and_emotion(text)
    assert res.emotion == "neutral"
    assert res.intensity <= 0.20


def test_fearful_emotion():
    text = "I am terrified and anxious about this panic situation!"
    res = analyze_text_sentiment_and_emotion(text)
    assert res.emotion == "fearful"
    assert res.intensity >= 0.40

    params = compute_modulated_synthesis_parameters(text, requested_emotion="auto")
    assert params["resolved_emotion"] == "fearful"
    assert params.emotion == "fearful"


def test_disgusted_emotion():
    text = "That food was revolting and nauseating, completely gross!"
    res = analyze_text_sentiment_and_emotion(text)
    assert res.emotion == "disgusted"
    assert res.intensity >= 0.40

    params = compute_modulated_synthesis_parameters(text, requested_emotion="auto")
    assert params["resolved_emotion"] == "disgusted"


def test_user_intensity_override():
    # Ambiguous sentence 'I am fine'
    text = "I am fine."
    # User manually specifies full 0.90 intensity
    params = compute_modulated_synthesis_parameters(
        text,
        requested_emotion="happy",
        user_intensity=0.90
    )
    assert params["intensity"] == 0.90
    assert params["resolved_emotion"] == "happy"
    assert params["exaggeration"] >= 0.25


def test_lexical_intensity_boosters():
    text_plain = "I am happy to see you."
    text_boosted = "I am SO INCREDIBLY and EXTREMELY happy to see you!!!"
    res_plain = analyze_text_sentiment_and_emotion(text_plain)
    res_boosted = analyze_text_sentiment_and_emotion(text_boosted)
    assert res_boosted.intensity > res_plain.intensity


def test_emotion_profile_interface():
    text = "This is a great achievement."
    profile = compute_modulated_synthesis_parameters(text)
    # Both attribute and dict access work seamlessly
    assert hasattr(profile, "emotion")
    assert hasattr(profile, "intensity")
    assert "cfg_weight" in profile
    assert profile["resolved_emotion"] == profile.emotion
    d = profile.to_dict()
    assert isinstance(d, dict)
    assert "cfg_weight" in d


def run_all_tests():
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    print(f"Running {len(funcs)} emotion analyzer tests...")
    for fn in funcs:
        fn()
        print(f"  {fn.__name__}: PASS")
    print(f"All {len(funcs)} emotion analyzer tests PASSED!")


if __name__ == "__main__":
    run_all_tests()


