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
- Discovers Chinese voices exposed by the current browser and operating system.
- Offers an explicit voice selector in every completed answer.
- Ranks natural, neural, online, Xiaoxiao, and Yunxi voices ahead of legacy
  desktop voices when `Auto (recommended)` is selected.
- Persists the user's voice choice in browser `localStorage`.
- Markdown markers are removed before playback.
- Playback can be stopped from the same control.

The available list depends on the browser and installed system voices. Chrome
and Edge may expose different sets of legacy and natural voices on the same
machine. MedQA does not download or bundle third-party speech models.

## Privacy

Browser speech recognition may use a browser vendor's online recognition
service. The implementation does not send audio to the MedQA FastAPI service.
Users should avoid speaking personal medical identifiers or other sensitive
information.

Playback is initiated in the browser. Installed local voices synthesize speech
on-device, while voices marked `online` may send the answer text to the browser
vendor's speech service. Do not read answers containing personal medical data
with an online voice when that transfer is not acceptable.

## Browser Support

Chrome and Edge provide the broadest support. Firefox and some embedded
browsers may not expose the recognition API; in that case MedQA keeps the
text-only workflow available and disables the microphone button.
