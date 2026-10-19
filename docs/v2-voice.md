# MedQA v2 Voice Interaction

MedQA v2 adds browser-native voice input and answer playback without adding a
server-side audio dependency.

## Voice Input

- Uses the Web Speech API (`SpeechRecognition` or `webkitSpeechRecognition`).
- Language is fixed to Mandarin Chinese (`zh-CN`).
- Interim recognition results populate the question composer while the user
  speaks.
- The microphone control is disabled when the browser does not support speech
  recognition.
- Permission, audio-device, no-speech, and network errors are shown inline.

## Answer Playback

- Uses `speechSynthesis` and `SpeechSynthesisUtterance`.
- Prefers an installed Chinese voice, with Mandarin as the fallback language.
- Markdown markers are removed before playback.
- Playback can be stopped from the same control.

## Privacy

Browser speech recognition may use a browser vendor's online recognition
service. The implementation does not send audio to the MedQA FastAPI service.
Users should avoid speaking personal medical identifiers or other sensitive
information. Answer playback happens locally in the browser.

## Browser Support

Chrome and Edge provide the broadest support. Firefox and some embedded
browsers may not expose the recognition API; in that case MedQA keeps the
text-only workflow available and disables the microphone button.
