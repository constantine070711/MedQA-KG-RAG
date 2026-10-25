"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

interface SpeechResultAlternative {
  transcript: string;
}

interface SpeechResult {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechResultAlternative;
}

interface SpeechResultList {
  length: number;
  [index: number]: SpeechResult;
}

interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: SpeechResultList;
}

interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

export interface SpeechVoiceOption {
  voiceURI: string;
  name: string;
  lang: string;
  localService: boolean;
  default: boolean;
}

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

const SPEECH_ERRORS: Record<string, string> = {
  "not-allowed": "未获得麦克风权限。",
  "service-not-allowed": "浏览器禁止了语音识别服务。",
  "audio-capture": "未检测到可用的麦克风。",
  network: "语音识别网络连接失败。",
  "no-speech": "没有检测到语音，请重试。",
};

const VOICE_STORAGE_KEY = "medqa:speech-voice";

function subscribeToBrowserCapability() {
  return () => {};
}

function getRecognitionSupport() {
  return Boolean(window.SpeechRecognition ?? window.webkitSpeechRecognition);
}

function getSynthesisSupport() {
  return (
    "speechSynthesis" in window && "SpeechSynthesisUtterance" in window
  );
}

function isChineseVoice(voice: SpeechSynthesisVoice) {
  return (
    /^zh(?:-|_)/i.test(voice.lang) ||
    /chinese|mandarin|中文|普通话/i.test(voice.name)
  );
}

function voiceQualityScore(voice: SpeechSynthesisVoice) {
  let score = 0;
  if (/natural/i.test(voice.name)) {
    score += 80;
  }
  if (/neural/i.test(voice.name)) {
    score += 60;
  }
  if (/online/i.test(voice.name)) {
    score += 40;
  }
  if (/xiaoxiao|xiaoyi|yunxi|yunjian|xiaohan|晓晓|晓伊|云希|云健/i.test(voice.name)) {
    score += 50;
  }
  if (/google/i.test(voice.name)) {
    score += 30;
  }
  if (!voice.localService) {
    score += 10;
  }
  if (voice.default) {
    score += 2;
  }
  if (/desktop|legacy/i.test(voice.name)) {
    score -= 20;
  }
  return score;
}

function chineseVoiceOptions(): SpeechVoiceOption[] {
  if (!("speechSynthesis" in window)) {
    return [];
  }
  return window.speechSynthesis
    .getVoices()
    .filter(isChineseVoice)
    .sort(
      (left, right) =>
        voiceQualityScore(right) - voiceQualityScore(left) ||
        left.name.localeCompare(right.name, "zh-CN"),
    )
    .map((voice) => ({
      voiceURI: voice.voiceURI,
      name: voice.name,
      lang: voice.lang,
      localService: voice.localService,
      default: voice.default,
    }));
}

function preferredChineseVoice(voices: SpeechSynthesisVoice[]) {
  return voices.filter(isChineseVoice).sort(
    (left, right) =>
      voiceQualityScore(right) - voiceQualityScore(left) ||
      left.name.localeCompare(right.name, "zh-CN"),
  )[0];
}

export function useSpeechInput(
  onTranscript: (text: string, isFinal: boolean) => void,
) {
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const onTranscriptRef = useRef(onTranscript);
  const supported = useSyncExternalStore(
    subscribeToBrowserCapability,
    getRecognitionSupport,
    () => false,
  );

  useEffect(() => {
    onTranscriptRef.current = onTranscript;
  }, [onTranscript]);

  useEffect(() => {
    return () => {
      recognitionRef.current?.abort();
      recognitionRef.current = null;
    };
  }, []);

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
    setListening(false);
  }, []);

  const start = useCallback(() => {
    const Recognition =
      window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!Recognition) {
      setError("当前浏览器不支持语音输入。");
      return;
    }

    recognitionRef.current?.abort();
    const recognition = new Recognition();
    recognition.lang = "zh-CN";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const transcript = result[0]?.transcript ?? "";
        if (result.isFinal) {
          finalText += transcript;
        } else {
          interimText += transcript;
        }
      }
      const text = (finalText || interimText).trim();
      if (text) {
        onTranscriptRef.current(text, Boolean(finalText));
      }
    };
    recognition.onerror = (event) => {
      setError(SPEECH_ERRORS[event.error] ?? "语音输入失败，请重试。");
      setListening(false);
    };
    recognition.onend = () => {
      setListening(false);
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    setError(null);
    setListening(true);
    try {
      recognition.start();
    } catch {
      setListening(false);
      setError("语音输入启动失败，请稍后重试。");
    }
  }, []);

  const toggle = useCallback(() => {
    if (listening) {
      stop();
    } else {
      start();
    }
  }, [listening, start, stop]);

  return { supported, listening, error, start, stop, toggle };
}

function textForSpeech(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/[#>*_`|~-]/g, " ")
    .replace(/\[(.*?)\]\(.*?\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

export function useSpeechOutput(onEnd?: () => void) {
  const [speaking, setSpeaking] = useState(false);
  const [voices, setVoices] = useState<SpeechVoiceOption[]>([]);
  const [voiceURI, setVoiceURI] = useState("");
  const onEndRef = useRef(onEnd);
  const supported = useSyncExternalStore(
    subscribeToBrowserCapability,
    getSynthesisSupport,
    () => false,
  );

  useEffect(() => {
    onEndRef.current = onEnd;
  }, [onEnd]);

  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel();
    };
  }, []);

  useEffect(() => {
    if (!("speechSynthesis" in window)) {
      return;
    }

    const loadVoices = () => {
      const nextVoices = chineseVoiceOptions();
      setVoices(nextVoices);
      setVoiceURI((current) => {
        if (current && nextVoices.some((voice) => voice.voiceURI === current)) {
          return current;
        }
        try {
          const stored = window.localStorage.getItem(VOICE_STORAGE_KEY);
          if (stored && nextVoices.some((voice) => voice.voiceURI === stored)) {
            return stored;
          }
        } catch {
          // Speech still works when storage is unavailable.
        }
        return "";
      });
    };

    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", loadVoices);
    };
  }, []);

  const selectVoice = useCallback((nextVoiceURI: string) => {
    setVoiceURI(nextVoiceURI);
    try {
      if (nextVoiceURI) {
        window.localStorage.setItem(VOICE_STORAGE_KEY, nextVoiceURI);
      } else {
        window.localStorage.removeItem(VOICE_STORAGE_KEY);
      }
    } catch {
      // Ignore storage failures and keep the in-memory selection.
    }
  }, []);

  const stop = useCallback(() => {
    window.speechSynthesis?.cancel();
    setSpeaking(false);
    onEndRef.current?.();
  }, []);

  const speak = useCallback((text: string) => {
    if (!("speechSynthesis" in window)) {
      return;
    }
    const cleaned = textForSpeech(text);
    if (!cleaned) {
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(cleaned);
    utterance.lang = "zh-CN";
    utterance.rate = 0.98;
    utterance.pitch = 1;
    const availableVoices = window.speechSynthesis.getVoices();
    const selectedVoice = voiceURI
      ? availableVoices.find((voice) => voice.voiceURI === voiceURI)
      : undefined;
    const outputVoice = selectedVoice ?? preferredChineseVoice(availableVoices);
    if (outputVoice) {
      utterance.voice = outputVoice;
      utterance.lang = outputVoice.lang;
    }
    utterance.onend = () => {
      setSpeaking(false);
      onEndRef.current?.();
    };
    utterance.onerror = () => {
      setSpeaking(false);
      onEndRef.current?.();
    };
    setSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }, [voiceURI]);

  return {
    supported,
    speaking,
    voices,
    voiceURI,
    setVoiceURI: selectVoice,
    speak,
    stop,
  };
}
