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
    const chineseVoice = window.speechSynthesis
      .getVoices()
      .find((voice) => voice.lang.toLowerCase().startsWith("zh"));
    if (chineseVoice) {
      utterance.voice = chineseVoice;
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
  }, []);

  return { supported, speaking, speak, stop };
}
