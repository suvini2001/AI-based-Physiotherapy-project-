# audio_feedback.py
"""
Offline text-to-speech feedback in a background thread.
Does not block the video loop.
"""
import threading

_engine  = None
_lock    = threading.Lock()
_busy    = False


def _init():
    global _engine
    try:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 145)
        _engine.setProperty('volume', 1.0)
        print("Audio feedback: ready")
    except Exception as e:
        print(f"Audio feedback disabled ({e}). "
              f"Run: pip install pyttsx3")
        _engine = None


_init()


def speak(text, priority=False):
    """
    Fire-and-forget TTS.
    priority=True: interrupt current speech (for fault cues).
    priority=False: skip if already speaking (for rep count).
    """
    global _busy
    if _engine is None:
        return

    def _run():
        global _busy
        with _lock:
            _busy = True
            try:
                _engine.say(text)
                _engine.runAndWait()
            except Exception:
                pass
            finally:
                _busy = False

    if priority or not _busy:
        threading.Thread(target=_run, daemon=True).start()