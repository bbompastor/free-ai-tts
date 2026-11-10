/**
 * OpenVoice TTS Worker
 * Runs Kokoro-82M ONNX model client-side in an isolated Web Worker.
 * Supports US/UK English + European Portuguese (Portugal, pt-PT) with native client-side G2P IPA phonemization.
 * Handles automatic text chunking and seamless WAV concatenation.
 */
import { KokoroTTS } from "https://esm.sh/kokoro-js";

let ttsInstance = null;
let currentDtype = null;
let currentDevice = null;

const STYLE_DIM = 256;
const VOICE_URLS = [
  "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices",
  "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped/resolve/main/voices"
];

// Map European Portuguese (pt-PT) voice IDs to Kokoro v1.0 Portuguese voice embedding files
const PORTUGUESE_VOICE_MAP = {
  pt_dora: { bin: "pf_dora", lang: "pt-PT" },
  pt_alex: { bin: "pm_alex", lang: "pt-PT" },
  pt_santa: { bin: "pm_santa", lang: "pt-PT" },
  pf_dora: { bin: "pf_dora", lang: "pt-PT" },
  pm_alex: { bin: "pm_alex", lang: "pt-PT" },
  pm_santa: { bin: "pm_santa", lang: "pt-PT" }
};

const customVoiceMemoryCache = new Map();

/**
 * Fetches and caches a 256-dim Kokoro voice style binary (.bin) for multilingual voices.
 */
async function getCustomVoiceData(binName) {
  if (customVoiceMemoryCache.has(binName)) {
    return customVoiceMemoryCache.get(binName);
  }

  let cache = null;
  try {
    if (typeof caches !== "undefined") {
      cache = await caches.open("kokoro-voices-v1");
    }
  } catch (e) {
    // Ignore Cache Storage errors in restricted contexts
  }

  for (const baseUrl of VOICE_URLS) {
    const url = `${baseUrl}/${binName}.bin`;
    try {
      let response = cache ? await cache.match(url) : null;
      if (!response) {
        response = await fetch(url);
        if (response.ok && cache) {
          await cache.put(url, response.clone());
        }
      }
      if (response && response.ok) {
        const buffer = await response.arrayBuffer();
        const floatArray = new Float32Array(buffer);
        customVoiceMemoryCache.set(binName, floatArray);
        return floatArray;
      }
    } catch (err) {
      console.warn(`Voice fetch attempt failed for ${url}:`, err);
    }
  }

  throw new Error(`Could not load Portuguese voice embedding (${binName}.bin).`);
}

/**
 * High-frequency European Portuguese (pt-PT, Lisbon/Coimbra standard) lexicon
 * for function words, irregular open/closed mid-vowels, and common vocabulary.
 */
const PT_PT_LEXICON = {
  // Function words & clitics (unstressed or lightly stressed in pt-PT)
  "o": "u", "a": "ɐ", "os": "uʃ", "as": "ɐʃ",
  "um": "ˈũ", "uma": "ˈumɐ", "uns": "ˈũʃ", "umas": "ˈumɐʃ",
  "de": "dɨ", "do": "du", "da": "dɐ", "dos": "duʃ", "das": "dɐʃ",
  "dum": "dˈũ", "duma": "dˈumɐ",
  "em": "ɐ̃j̃", "no": "nu", "na": "nɐ", "nos": "nuʃ", "nas": "nɐʃ",
  "num": "nˈũ", "numa": "nˈumɐ", "nuns": "nˈũʃ", "numas": "nˈumɐʃ",
  "por": "puɾ", "pelo": "pɨlu", "pela": "pɨlɐ", "pelos": "pɨluʃ", "pelas": "pɨlɐʃ",
  "para": "pˈɐɾɐ", "pra": "pɾɐ", "pro": "pɾu",
  "com": "kˈõ", "sem": "sˈɐ̃j̃", "sob": "sˈɔb", "sobre": "sˈobɾɨ", "entre": "ˈẽtɾɨ",
  "até": "ɐtˈɛ", "desde": "dˈeʒdɨ", "contra": "kˈõtɾɐ", "perante": "pɨɾˈɐ̃tɨ",
  "e": "i", "ou": "o", "mas": "mɐʃ", "nem": "nˈɐ̃j̃", "que": "kɨ", "se": "sɨ",
  "como": "kˈomu", "quando": "kwˈɐ̃du", "onde": "ˈõdɨ", "porque": "puɾkˈɨ", "porquê": "puɾkˈe",
  "quem": "kˈɐ̃j̃", "qual": "kwˈal", "quais": "kwˈajʃ", "quanto": "kwˈɐ̃tu", "quantos": "kwˈɐ̃tuʃ",
  "ao": "aw", "aos": "awʃ", "à": "a", "às": "aʃ",
  // Pronouns & determiners
  "eu": "ˈew", "tu": "tˈu", "ele": "ˈelɨ", "ela": "ˈɛlɐ", "eles": "ˈelɨʃ", "elas": "ˈɛlɐʃ",
  "nós": "nˈɔʃ", "vós": "vˈɔʃ", "você": "vusˈe", "vocês": "vusˈeʃ",
  "me": "mɨ", "te": "tɨ", "lhe": "ʎɨ", "lhes": "ʎɨʃ",
  "mim": "mˈĩ", "ti": "tˈi", "si": "sˈi", "comigo": "kumˈiɡu", "contigo": "kõtˈiɡu", "consigo": "kõsˈiɡu",
  "meu": "mˈew", "minha": "mˈiɲɐ", "meus": "mˈewʃ", "minhas": "mˈiɲɐʃ",
  "teu": "tˈew", "tua": "tˈuɐ", "teus": "tˈewʃ", "tuas": "tˈuɐʃ",
  "seu": "sˈew", "sua": "sˈuɐ", "seus": "sˈewʃ", "suas": "sˈuɐʃ",
  "nosso": "nˈɔsu", "nossa": "nˈɔsɐ", "nossos": "nˈɔsuʃ", "nossas": "nˈɔsɐʃ",
  "vosso": "vˈɔsu", "vossa": "vˈɔsɐ", "vossos": "vˈɔsuʃ", "vossas": "vˈɔsɐʃ",
  "este": "ˈeʃtɨ", "esta": "ˈɛʃtɐ", "estes": "ˈeʃtɨʃ", "estas": "ˈɛʃtɐʃ", "isto": "ˈiʃtu",
  "esse": "ˈesɨ", "essa": "ˈɛsɐ", "esses": "ˈesɨʃ", "essas": "ˈɛsɐʃ", "isso": "ˈisu",
  "aquele": "ɐkˈelɨ", "aquela": "ɐkˈɛlɐ", "aqueles": "ɐkˈelɨʃ", "aquelas": "ɐkˈɛlɐʃ", "aquilo": "ɐkˈilu",
  "todo": "tˈodu", "toda": "tˈodɐ", "todos": "tˈoduʃ", "todas": "tˈodɐʃ", "tudo": "tˈudu",
  "outro": "ˈotɾu", "outra": "ˈotɾɐ", "outros": "ˈotɾuʃ", "outras": "ˈotɾɐʃ",
  "mesmo": "mˈeʒmu", "mesma": "mˈeʒmɐ", "mesmos": "mˈeʒmuʃ", "mesmas": "mˈeʒmɐʃ",
  "muito": "mˈũj̃tu", "muita": "mˈũj̃tɐ", "muitos": "mˈũj̃tuʃ", "muitas": "mˈũj̃tɐʃ",
  "pouco": "pˈoku", "pouca": "pˈokɐ", "poucos": "pˈokuʃ", "poucas": "pˈokɐʃ",
  "mais": "mˈajʃ", "menos": "mˈenuʃ", "tão": "tˈɐ̃w̃", "tanto": "tˈɐ̃tu", "tanta": "tˈɐ̃tɐ",
  "algum": "aɫɡˈũ", "alguma": "aɫɡˈumɐ", "alguns": "aɫɡˈũʃ", "algumas": "aɫɡˈumɐʃ", "alguém": "aɫɡˈɐ̃j̃",
  "nenhum": "nɨɲˈũ", "nenhuma": "nɨɲˈumɐ", "ninguém": "nĩɡˈɐ̃j̃", "nada": "nˈadɐ", "cada": "kˈadɐ",
  // Common verbs & irregular vowel apertures
  "ser": "sˈeɾ", "sou": "sˈo", "és": "ˈɛʃ", "é": "ˈɛ", "somos": "sˈomuʃ", "são": "sˈɐ̃w̃",
  "era": "ˈɛɾɐ", "eram": "ˈɛɾɐ̃w̃", "foi": "fˈoj", "foram": "fˈoɾɐ̃w̃", "seja": "sˈɐjʒɐ",
  "estar": "iʃtˈaɾ", "estou": "iʃtˈo", "estás": "iʃtˈaʃ", "está": "iʃtˈa", "estamos": "iʃtˈɐmuʃ", "estão": "iʃtˈɐ̃w̃",
  "ter": "tˈeɾ", "tenho": "tˈɐɲu", "tens": "tˈɐ̃j̃ʃ", "tem": "tˈɐ̃j̃", "temos": "tˈemuʃ", "têm": "tˈɐ̃j̃ɐ̃j̃",
  "tinha": "tˈiɲɐ", "teve": "tˈevɨ", "tiveram": "tivˈɛɾɐ̃w̃",
  "haver": "ɐvˈeɾ", "há": "ˈa", "havia": "ɐvˈiɐ", "houve": "ˈovɨ",
  "ir": "ˈiɾ", "vou": "vˈo", "vais": "vˈajʃ", "vai": "vˈaj", "vamos": "vˈɐmuʃ", "vão": "vˈɐ̃w̃",
  "vir": "vˈiɾ", "venho": "vˈɐɲu", "vens": "vˈɐ̃j̃ʃ", "vem": "vˈɐ̃j̃", "vimos": "vˈimuʃ", "vêm": "vˈɐ̃j̃ɐ̃j̃",
  "fazer": "fɐzˈeɾ", "faço": "fˈasu", "faz": "fˈaʃ", "fez": "fˈeʃ", "feito": "fˈɐjtu",
  "dizer": "dizˈeɾ", "digo": "dˈiɡu", "diz": "dˈiʃ", "disse": "dˈisɨ", "dito": "dˈitu",
  "poder": "pudˈeɾ", "posso": "pˈɔsu", "pode": "pˈɔdɨ", "podem": "pˈɔdɐ̃j̃", "pôde": "pˈodɨ",
  "querer": "kɨɾˈeɾ", "quero": "kˈɛɾu", "quer": "kˈɛɾ", "querem": "kˈɛɾɐ̃j̃", "quis": "kˈiʃ",
  "saber": "sɐbˈeɾ", "sei": "sˈɐj", "sabe": "sˈabɨ", "sabem": "sˈabɐ̃j̃", "soube": "sˈobɨ",
  "ver": "vˈeɾ", "vejo": "vˈɐjʒu", "vê": "vˈe", "visto": "vˈiʃtu", "viu": "vˈiw",
  "dar": "dˈaɾ", "dou": "dˈo", "dá": "dˈa", "dão": "dˈɐ̃w̃", "deu": "dˈew",
  "pôr": "pˈoɾ", "ponho": "pˈoɲu", "põe": "pˈõj̃", "põem": "pˈõj̃ɐ̃j̃", "pôs": "pˈoʃ", "posto": "pˈoʃtu",
  // Common nouns, adjectives & adverbs with specific PT-PT pronunciation
  "não": "nˈɐ̃w̃", "sim": "sˈĩ", "bem": "bˈɐ̃j̃", "mal": "mˈal",
  "bom": "bˈõ", "boa": "bˈoɐ", "bons": "bˈõʃ", "boas": "bˈoɐʃ",
  "mau": "mˈaw", "má": "mˈa", "maus": "mˈawʃ", "más": "mˈaʃ",
  "maior": "mɐjˈɔɾ", "menor": "mɨnˈɔɾ", "melhor": "mɨʎˈɔɾ", "pior": "piˈɔɾ",
  "novo": "nˈovu", "nova": "nˈɔvɐ", "novos": "nˈɔvuʃ", "novas": "nˈɔvɐʃ",
  "povo": "pˈovu", "povos": "pˈɔvuʃ", "fogo": "fˈoɡu", "jogo": "ʒˈoɡu", "jogos": "ʒˈɔɡuʃ",
  "olho": "ˈoʎu", "olhos": "ˈɔʎuʃ", "ovo": "ˈovu", "ovos": "ˈɔvuʃ",
  "hora": "ˈɔɾɐ", "horas": "ˈɔɾɐʃ", "agora": "ɐɡˈɔɾɐ", "fora": "fˈɔɾɐ", "embora": "ĩbˈɔɾɐ", "ocorre": "ɔkˈɔʁɨ",
  "hoje": "ˈoʒɨ", "ontem": "ˈõtɐ̃j̃", "amanhã": "ɐmɐɲˈɐ̃", "sempre": "sˈẽpɾɨ", "nunca": "nˈũkɐ",
  "já": "ʒˈa", "ainda": "ɐˈĩdɐ", "também": "tɐ̃bˈɐ̃j̃", "assim": "ɐsˈĩ", "aqui": "ɐkˈi", "ali": "ɐlˈi",
  "portugal": "puɾtuɡˈal", "português": "puɾtuɡˈeʃ", "portuguesa": "puɾtuɡˈezɐ",
  "portugueses": "puɾtuɡˈezɨʃ", "portuguesas": "puɾtuɡˈezɐʃ",
  "lisboa": "liʒbˈoɐ", "porto": "pˈoɾtu", "coimbra": "kuˈĩbɾɐ",
  "olá": "ɔlˈa", "adeus": "ɐdˈewʃ", "obrigado": "ɔbɾiɡˈadu", "obrigada": "ɔbɾiɡˈadɐ",
  "voz": "vˈɔʃ", "vozes": "vˈɔzɨʃ", "vez": "vˈeʃ", "vezes": "vˈezɨʃ",
  "luz": "lˈuʃ", "paz": "pˈaʃ", "cruz": "kɾˈuʃ", "raiz": "ʁɐˈiʃ",
  "mês": "mˈeʃ", "meses": "mˈezɨʃ", "três": "tɾˈeʃ", "dez": "dˈɛʃ", "seis": "sˈɐjʃ", "dois": "dˈojʃ", "duas": "dˈuɐʃ",
  "país": "pɐˈiʃ", "países": "pɐˈizɨʃ", "pessoas": "pɨsˈoɐʃ", "pessoa": "pɨsˈoɐ",
  "tempo": "tˈẽpu", "mundo": "mˈũdu", "vida": "vˈidɐ", "dia": "dˈiɐ", "dias": "dˈiɐʃ",
  "noite": "nˈojtɨ", "tarde": "tˈaɾdɨ", "manhã": "mɐɲˈɐ̃",
  "texto": "tˈeʃtu", "textos": "tˈeʃtuʃ", "exemplo": "izˈẽplu", "exemplos": "izˈẽpluʃ",
  "exame": "izˈɐmɨ", "exames": "izˈɐmɨʃ", "êxito": "ˈezitu",
  "próximo": "pɾˈɔsimu", "próxima": "pɾˈɔsimɐ", "próximos": "pɾˈɔsimuʃ", "próximas": "pɾˈɔsimɐʃ",
  "máximo": "mˈasimu", "máxima": "mˈasimɐ", "mínimo": "mˈinimu", "mínima": "mˈinimɐ",
  "auxílio": "awsˈilju", "táxi": "tˈaksi", "fixo": "fˈiksu", "fixa": "fˈiksɐ",
  "complexo": "kõplˈɛksu", "complexa": "kõplˈɛksɐ", "fluxo": "flˈuksu",
  "puretts": "pˈuɾɨ tˈe tˈe ˈɛsɨ",
  "tts": "tˈe tˈe ˈɛsɨ",
  "ai": "ˈaj", "ia": "ˈi ˈa", "pdf": "pˈe dˈe ˈɛfɨ", "mp3": "ˈɛmɨ pˈe tɾˈeʃ", "wav": "wˈav",
  "webgpu": "wˈɛb ʒˈe pˈe ˈu", "cpu": "sˈe pˈe ˈu", "gpu": "ʒˈe pˈe ˈu"
};

const PT_UNITS = ["zˈɛɾu", "ˈũ", "dˈojʃ", "tɾˈeʃ", "kwˈatɾu", "sˈĩku", "sˈɐjʃ", "sˈɛtɨ", "ˈojtu", "nˈɔvɨ"];
const PT_TEENS = ["dˈɛʃ", "ˈõzɨ", "dˈozɨ", "tɾˈezɨ", "kɐtˈoɾzɨ", "kˈĩzɨ", "dɨzɐsˈɐjʃ", "dɨzɐsˈɛtɨ", "dɨzˈojtu", "dɨzɐnˈɔvɨ"];
const PT_TENS = ["", "", "vˈĩtɨ", "tɾˈĩtɐ", "kwɐɾˈẽtɐ", "sĩkwˈẽtɐ", "sɨsˈẽtɐ", "sɨtˈẽtɐ", "ojtˈẽtɐ", "nuvˈẽtɐ"];
const PT_HUNDREDS = ["", "sˈẽtu", "duzˈẽtuʃ", "tɾɨzˈẽtuʃ", "kwatɾusˈẽtuʃ", "kiɲˈẽtuʃ", "sɐjʃsˈẽtuʃ", "sɛtɨsˈẽtuʃ", "ojtusˈẽtuʃ", "nɔvɨsˈẽtuʃ"];

/**
 * Converts an integer (0 - 999999) into European Portuguese IPA phonemes.
 */
function numberToPortugueseIPA(n) {
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 10) return PT_UNITS[n];
  if (n < 20) return PT_TEENS[n - 10];
  if (n < 100) {
    const t = Math.floor(n / 10);
    const u = n % 10;
    return u === 0 ? PT_TENS[t] : `${PT_TENS[t]} i ${PT_UNITS[u]}`;
  }
  if (n === 100) return "sˈɐ̃j̃";
  if (n < 1000) {
    const h = Math.floor(n / 100);
    const rem = n % 100;
    return rem === 0 ? PT_HUNDREDS[h] : `${PT_HUNDREDS[h]} i ${numberToPortugueseIPA(rem)}`;
  }
  if (n < 1000000) {
    const th = Math.floor(n / 1000);
    const rem = n % 1000;
    const thStr = th === 1 ? "mˈil" : `${numberToPortugueseIPA(th)} mˈil`;
    if (rem === 0) return thStr;
    const conj = rem < 100 || rem % 100 === 0 ? " i " : " ";
    return `${thStr}${conj}${numberToPortugueseIPA(rem)}`;
  }
  return String(n);
}

/**
 * Rule-based European Portuguese (pt-PT) word-to-IPA converter.
 * Implements standard Portugal phonology:
 * - Unstressed vowel reduction (a -> ɐ, e -> ɨ / initial i, o -> u)
 * - Syllable-final sibilant palatalization ("chiado": coda s/z -> ʃ)
 * - Non-palatalized dental stops (/t/ and /d/ before /i/ and /ɨ/)
 * - European Portuguese -ei- -> ɐj and -em/-ens -> ɐ̃j̃ / ɐ̃j̃ʃ
 */
function phonemizePortugueseWord(rawWord) {
  const w = rawWord.toLowerCase().trim();
  if (!w) return "";
  if (PT_PT_LEXICON[w]) return PT_PT_LEXICON[w];

  // Numeric strings
  if (/^\d+$/.test(w)) {
    return numberToPortugueseIPA(parseInt(w, 10));
  }

  // Handle hyphenated compounds recursively (e.g. "bem-vindo")
  if (w.includes("-")) {
    return w
      .split("-")
      .filter(Boolean)
      .map(part => phonemizePortugueseWord(part))
      .join(" ");
  }

  const chars = Array.from(w);
  const len = chars.length;
  const isVowelChar = (c) => Boolean(c) && "aáàâãeéêiíoóôõuú".includes(c);
  const hasAccent = (c) => Boolean(c) && "áàâãéêíóôõú".includes(c);

  // 1. Identify vowel nuclei indices to determine primary stress position in Portuguese
  const vowelIndices = [];
  for (let i = 0; i < len; i++) {
    const c = chars[i];
    const prev = i > 0 ? chars[i - 1] : "";
    const next = i + 1 < len ? chars[i + 1] : "";
    // Skip silent 'u' in qu/gu before e/i
    if (c === "u" && (prev === "q" || prev === "g") && (next === "e" || next === "é" || next === "ê" || next === "i" || next === "í")) {
      continue;
    }
    if (isVowelChar(c)) {
      vowelIndices.push(i);
    }
  }

  if (vowelIndices.length === 0) {
    return w;
  }

  // Group adjacent vowels into syllable nuclei (diphthongs vs hiatus)
  const nuclei = [];
  let idx = 0;
  while (idx < vowelIndices.length) {
    const vIdx = vowelIndices[idx];
    const c = chars[vIdx];
    const nextVIdx = idx + 1 < vowelIndices.length ? vowelIndices[idx + 1] : -1;
    if (nextVIdx === vIdx + 1) {
      const nextC = chars[nextVIdx];
      // Falling diphthongs or nasal diphthongs stay in one syllable nucleus unless second vowel has an acute/circumflex accent
      const formsDiphthong =
        !hasAccent(nextC) &&
        (("aáâãeéêoóôõu".includes(c) && (nextC === "i" || nextC === "u")) ||
          (c === "ã" && (nextC === "e" || nextC === "o")) ||
          (c === "õ" && nextC === "e") ||
          (c === "i" && nextC === "u") ||
          (c === "u" && nextC === "i"));
      if (formsDiphthong) {
        nuclei.push({ start: vIdx, end: nextVIdx });
        idx += 2;
        continue;
      }
    }
    nuclei.push({ start: vIdx, end: vIdx });
    idx++;
  }

  // Determine which nucleus carries primary stress
  let stressedNucleusIdx = -1;
  for (let n = nuclei.length - 1; n >= 0; n--) {
    const nuc = nuclei[n];
    if (hasAccent(chars[nuc.start]) || hasAccent(chars[nuc.end])) {
      stressedNucleusIdx = n;
      break;
    }
  }

  if (stressedNucleusIdx === -1) {
    if (nuclei.length === 1) {
      stressedNucleusIdx = 0;
    } else {
      // Portuguese default stress rule:
      // Paroxytone (penultimate) if ending in -a(s), -e(s), -o(s), -am, -em, -ens
      // Oxytone (final) if ending in -i(s), -u(s), -r, -l, -z, -x, -im, -ins, -um, -uns, -om, -ons, or diphthong
      const lastNuc = nuclei[nuclei.length - 1];
      const endsInDiphthong = lastNuc.end > lastNuc.start;
      if (!endsInDiphthong && /(?:[aeo]s?|am|em|ens)$/.test(w)) {
        stressedNucleusIdx = nuclei.length - 2;
      } else {
        stressedNucleusIdx = nuclei.length - 1;
      }
    }
  }

  const stressedVowelPos = nuclei[stressedNucleusIdx].start;

  // 2. Scan graphemes left-to-right and emit European Portuguese IPA
  let ipa = "";
  let i = 0;
  while (i < len) {
    const c = chars[i];
    const prev = i > 0 ? chars[i - 1] : "";
    const next = i + 1 < len ? chars[i + 1] : "";
    const next2 = i + 2 < len ? chars[i + 2] : "";
    const isStressed = i === stressedVowelPos;
    const stressMark = isStressed ? "ˈ" : "";

    // Multi-character consonant digraphs
    if (c === "l" && next === "h") {
      ipa += "ʎ";
      i += 2;
      continue;
    }
    if (c === "n" && next === "h") {
      ipa += "ɲ";
      i += 2;
      continue;
    }
    if (c === "c" && next === "h") {
      ipa += "ʃ";
      i += 2;
      continue;
    }
    if (c === "r" && next === "r") {
      ipa += "ʁ";
      i += 2;
      continue;
    }
    if (c === "s" && next === "s") {
      ipa += "s";
      i += 2;
      continue;
    }
    if (c === "s" && next === "c" && (next2 === "e" || next2 === "é" || next2 === "ê" || next2 === "i" || next2 === "í")) {
      ipa += "ʃs";
      i += 2;
      continue;
    }
    if (c === "q" && next === "u") {
      if (next2 === "e" || next2 === "é" || next2 === "ê" || next2 === "i" || next2 === "í") {
        ipa += "k";
        i += 2;
      } else {
        ipa += "kw";
        i += 2;
      }
      continue;
    }
    if (c === "g" && next === "u") {
      if (next2 === "e" || next2 === "é" || next2 === "ê" || next2 === "i" || next2 === "í") {
        ipa += "ɡ";
        i += 2;
      } else {
        ipa += "ɡw";
        i += 2;
      }
      continue;
    }

    // Nasal diphthongs with tilde (ão, ãos, ãe, ães, õe, ões)
    if (c === "ã" && next === "o") {
      ipa += `${stressMark}ɐ̃w̃`;
      i += 2;
      continue;
    }
    if (c === "ã" && next === "e") {
      ipa += `${stressMark}ɐ̃j̃`;
      i += 2;
      continue;
    }
    if (c === "õ" && next === "e") {
      ipa += `${stressMark}õj̃`;
      i += 2;
      continue;
    }

    // Word-final -em / -ens in European Portuguese -> ɐ̃j̃ / ɐ̃j̃ʃ (e.g. "viagem", "homens", "também")
    if ((c === "e" || c === "é" || c === "ê") && ((next === "m" && i + 2 === len) || (next === "n" && next2 === "s" && i + 3 === len))) {
      ipa += `${stressMark}ɐ̃j̃`;
      if (next === "n") {
        ipa += "ʃ";
        i += 3;
      } else {
        i += 2;
      }
      continue;
    }

    // Word-final -am (3rd person plural verbs like "falam", "ocorre" vs "ocorrem") -> ɐ̃w̃
    if (c === "a" && next === "m" && i + 2 === len) {
      ipa += `${stressMark}ɐ̃w̃`;
      i += 2;
      continue;
    }

    // Pre-consonantal nasal vowels (vowel + m/n before another consonant or word-end)
    if (isVowelChar(c) && (next === "m" || next === "n") && !isVowelChar(next2) && !(next === "n" && next2 === "h")) {
      if ("aáâã".includes(c)) ipa += `${stressMark}ɐ̃`;
      else if ("eéê".includes(c)) ipa += `${stressMark}ẽ`;
      else if ("ií".includes(c)) ipa += `${stressMark}ĩ`;
      else if ("oóôõ".includes(c)) ipa += `${stressMark}õ`;
      else if ("uú".includes(c)) ipa += `${stressMark}ũ`;
      i += 2;
      continue;
    }

    // Oral diphthongs (vowel + i/u in same nucleus)
    if (isVowelChar(c) && (next === "i" || next === "u") && !hasAccent(next)) {
      // European Portuguese "-ei-" centralized diphthong -> ɐj (e.g. "inteiramente", "primeiro", "leite")
      if ((c === "e" || c === "ê") && next === "i") {
        ipa += `${stressMark}ɐj`;
        i += 2;
        continue;
      }
      if ((c === "a" || c === "á") && next === "i") {
        ipa += `${stressMark}aj`;
        i += 2;
        continue;
      }
      if (c === "é" && next === "i") {
        ipa += `${stressMark}ɛj`;
        i += 2;
        continue;
      }
      if ((c === "o" || c === "ô") && next === "i") {
        ipa += `${stressMark}oj`;
        i += 2;
        continue;
      }
      if (c === "ó" && next === "i") {
        ipa += `${stressMark}ɔj`;
        i += 2;
        continue;
      }
      if ((c === "u" || c === "ú") && next === "i") {
        ipa += `${stressMark}uj`;
        i += 2;
        continue;
      }
      if ((c === "a" || c === "á") && next === "u") {
        ipa += `${stressMark}aw`;
        i += 2;
        continue;
      }
      if ((c === "e" || c === "ê") && next === "u") {
        ipa += `${stressMark}ew`;
        i += 2;
        continue;
      }
      if (c === "é" && next === "u") {
        ipa += `${stressMark}ɛw`;
        i += 2;
        continue;
      }
      if ((c === "o" || c === "ô") && next === "u") {
        // In European Portuguese, "ou" monophthongizes to /o/
        ipa += `${stressMark}o`;
        i += 2;
        continue;
      }
      if ((c === "i" || c === "í") && next === "u") {
        ipa += `${stressMark}iw`;
        i += 2;
        continue;
      }
    }

    // Rising diphthongs (unstressed i/u before a vowel in hiatus, e.g. "estúdio", "inteligência", "artificial")
    if ((c === "i" || c === "u") && !isStressed && isVowelChar(next)) {
      ipa += c === "i" ? "j" : "w";
      i++;
      continue;
    }

    // Single vowels (with European Portuguese vowel reduction when unstressed)
    if (isVowelChar(c)) {
      if (c === "á" || c === "à") {
        ipa += `${stressMark}a`;
      } else if (c === "â") {
        ipa += `${stressMark}ɐ`;
      } else if (c === "ã") {
        ipa += `${stressMark}ɐ̃`;
      } else if (c === "a") {
        if (isStressed) {
          // Stressed 'a' before intervocalic nasal m/n/nh is central /ɐ/ in PT-PT (e.g. "cama", "ano", "banho")
          const beforeNasal = (next === "m" || next === "n") && isVowelChar(next2);
          ipa += `${stressMark}${beforeNasal ? "ɐ" : "a"}`;
        } else {
          ipa += "ɐ";
        }
      } else if (c === "é") {
        ipa += `${stressMark}ɛ`;
      } else if (c === "ê") {
        ipa += `${stressMark}e`;
      } else if (c === "e") {
        if (isStressed) {
          // Stressed 'e' before palatal lh/nh/ch/j is centralized /ɐ/ in Lisbon PT-PT
          const beforePalatal = (next === "l" && next2 === "h") || (next === "n" && next2 === "h") || next === "j" || (next === "c" && next2 === "h");
          ipa += `${stressMark}${beforePalatal ? "ɐ" : "e"}`;
        } else if (i === 0 && (next === "s" || next === "z" || next === "x")) {
          // Initial "es-", "ex-" in PT-PT is /iʃ/, /iz/ (e.g. "estúdio" -> iʃtˈudju)
          ipa += "i";
        } else if (isVowelChar(next)) {
          ipa += "i";
        } else {
          // Signature European Portuguese unstressed 'e' -> /ɨ/
          ipa += "ɨ";
        }
      } else if (c === "í" || c === "i") {
        ipa += `${stressMark}i`;
      } else if (c === "ó") {
        ipa += `${stressMark}ɔ`;
      } else if (c === "ô") {
        ipa += `${stressMark}o`;
      } else if (c === "õ") {
        ipa += `${stressMark}õ`;
      } else if (c === "o") {
        if (isStressed) {
          ipa += `${stressMark}o`;
        } else if (i === 0 && next === "c" && next2 === "o") {
          ipa += "ɔ";
        } else {
          // Signature European Portuguese unstressed 'o' -> /u/
          ipa += "u";
        }
      } else if (c === "ú" || c === "u") {
        ipa += `${stressMark}u`;
      }
      i++;
      continue;
    }

    // Single consonants
    switch (c) {
      case "b":
      case "d":
      case "f":
      case "k":
      case "m":
      case "n":
      case "p":
      case "t":
      case "v":
        ipa += c;
        break;
      case "c":
        if (next === "e" || next === "é" || next === "ê" || next === "i" || next === "í") {
          ipa += "s";
        } else {
          ipa += "k";
        }
        break;
      case "ç":
        ipa += "s";
        break;
      case "g":
        if (next === "e" || next === "é" || next === "ê" || next === "i" || next === "í") {
          ipa += "ʒ";
        } else {
          ipa += "ɡ";
        }
        break;
      case "h":
        // Silent in Portuguese
        break;
      case "j":
        ipa += "ʒ";
        break;
      case "l":
        ipa += "l";
        break;
      case "q":
        ipa += "k";
        break;
      case "r":
        // Word-initial 'r' or 'r' after n/l/s is uvular /ʁ/; single intervocalic/coda 'r' is tap /ɾ/
        if (i === 0 || prev === "n" || prev === "l" || prev === "s") {
          ipa += "ʁ";
        } else {
          ipa += "ɾ";
        }
        break;
      case "s":
        if (i === 0) {
          ipa += "s";
        } else if (isVowelChar(prev) && isVowelChar(next)) {
          // Intervocalic 's' is voiced /z/
          ipa += "z";
        } else if (!isVowelChar(next)) {
          // European Portuguese coda 's' ("chiado"):
          // Voiced /ʒ/ before voiced consonants (b, d, g, v, m, n, l, r), voiceless /ʃ/ at word end or before voiceless consonants
          if (next && "bdgvmnlr".includes(next)) {
            ipa += "ʒ";
          } else {
            ipa += "ʃ";
          }
        } else {
          ipa += "s";
        }
        break;
      case "w":
        ipa += "w";
        break;
      case "x":
        if (i === 0) {
          ipa += "ʃ";
        } else if (i === 1 && prev === "e" && isVowelChar(next)) {
          // "ex-" + vowel -> /iz/ (e.g. "exato", "exame", "existir")
          ipa += "z";
        } else if (!isVowelChar(next)) {
          ipa += "ʃ";
        } else {
          ipa += "ʃ";
        }
        break;
      case "y":
        ipa += `${stressMark}i`;
        break;
      case "z":
        if (i === len - 1 || !isVowelChar(next)) {
          // Word-final or coda 'z' in PT-PT is /ʃ/ (e.g. "voz" -> vˈɔʃ, "luz" -> lˈuʃ)
          ipa += "ʃ";
        } else {
          ipa += "z";
        }
        break;
      default:
        break;
    }
    i++;
  }

  return ipa;
}

/**
 * Phonemizes text into European Portuguese (pt-PT) IPA phonemes while preserving punctuation
 * so Kokoro-82M maintains natural prosody, question intonation, and clause pauses.
 */
function phonemizePortugueseKeepPunctuation(text) {
  // Normalize percentage and currency symbols into spoken Portuguese words
  const normalized = text
    .replace(/(\d+)\s*%/g, "$1 por cento")
    .replace(/%/g, " por cento ")
    .replace(/€\s*(\d+)/g, "$1 euros")
    .replace(/(\d+)\s*€/g, "$1 euros");

  // Tokenize into words and punctuation clusters
  const tokens = normalized.match(/[A-Za-zÀ-ÖØ-öø-ÿ0-9-]+|[,;:.!?—…"«»“”]+/g) || [];
  const out = [];

  for (let t = 0; t < tokens.length; t++) {
    const tok = tokens[t];
    if (/^[,;:.!?—…"«»“”]+$/.test(tok)) {
      if (out.length > 0) {
        out[out.length - 1] += tok;
      }
    } else {
      const ph = phonemizePortugueseWord(tok);
      if (ph) {
        // Apply European Portuguese cross-word sibilant sandhi (liaison):
        // Word-final /ʃ/ (without intervening punctuation) becomes /z/ before a vowel-initial word
        // and /ʒ/ before a voiced consonant.
        if (out.length > 0 && out[out.length - 1].endsWith("ʃ")) {
          const nextFirst = ph.startsWith("ˈ") ? ph.slice(1, 2) : ph.slice(0, 1);
          if ("aɐeɛɨioɔujw".includes(nextFirst)) {
            out[out.length - 1] = out[out.length - 1].slice(0, -1) + "z";
          } else if ("bdɡvzʒmnɲlʎɾʁ".includes(nextFirst)) {
            out[out.length - 1] = out[out.length - 1].slice(0, -1) + "ʒ";
          }
        }
        out.push(ph);
      }
    }
  }

  return out.join(" ").replace(/\s+/g, " ").trim().normalize("NFD");
}

/**
 * Synthesizes a chunk of European Portuguese text using Kokoro-82M + eSpeak-NG pt-PT IPA phonemes.
 */
async function generatePortugueseChunk(tts, chunkTextStr, voiceId, speedValue) {
  const voiceCfg = PORTUGUESE_VOICE_MAP[voiceId];
  const phonemes = await phonemizePortugueseKeepPunctuation(chunkTextStr, voiceCfg.lang);
  if (!phonemes) {
    return { audio: new Float32Array(0), sampling_rate: 24000 };
  }

  const { input_ids } = tts.tokenizer(phonemes, { truncation: true });
  const seqLen = input_ids.dims ? input_ids.dims.at(-1) : input_ids.size;
  const numTokens = Math.min(Math.max(seqLen - 2, 0), 509);

  const voiceData = await getCustomVoiceData(voiceCfg.bin);
  const offset = numTokens * STYLE_DIM;
  const styleSlice = voiceData.slice(offset, offset + STYLE_DIM);

  const TensorCtor = input_ids.constructor;
  const inputs = {
    input_ids,
    style: new TensorCtor("float32", styleSlice, [1, STYLE_DIM]),
    speed: new TensorCtor("float32", [speedValue], [1])
  };

  const { waveform } = await tts.model(inputs);
  return {
    audio: waveform.data,
    sampling_rate: 24000
  };
}

/**
 * Initializes Kokoro TTS with the optimal precision for the target device:
 * - WebGPU: Enforces "fp32" on "onnx-community/Kokoro-82M-v1.0-ONNX".
 *   CRITICAL: ONNX Runtime WebGPU integer quantization (q8/q4) has known shader kernel bugs
 *   that cause garbled/robotic noise (q8) or accelerated/chipmunk speech (q4).
 *   Float32 runs natively on GPU tensor cores with 100% mathematical fidelity and crystal-clear human sound.
 * - WASM CPU: Uses "q8" (or "q4") which runs smoothly via CPU SIMD instructions with zero GPU driver bugs.
 */
async function getOrInitTTS(requestedMode = "webgpu") {
  const wantsWasm = requestedMode === "wasm" || requestedMode === "wasm-mobile" || requestedMode === "q8";
  const isMobileWebGPU = requestedMode === "webgpu-mobile";

  // Check if WebGPU is available in this worker environment
  let canUseWebGPU = false;
  if (!wantsWasm && typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: isMobileWebGPU ? "low-power" : "high-performance"
      });
      if (adapter) {
        canUseWebGPU = true;
      }
    } catch (e) {
      console.warn("Worker WebGPU requestAdapter failed:", e);
      canUseWebGPU = false;
    }
  }

  // WebGPU uses fp32 with Kokoro-82M-v1.0-ONNX for 100% mathematical fidelity and crystal-clear human sound.
  // On mobile WebGPU ("webgpu-mobile"), we pair it with low-power adapter hints and 110-char micro-chunking
  // so transient GPU activation buffers stay well under mobile OS per-tab memory limits.
  const targetDevice = canUseWebGPU ? "webgpu" : "wasm";
  const targetDtype = canUseWebGPU ? "fp32" : "q8";
  const cacheKey = `${targetDevice}-${targetDtype}`;

  if (ttsInstance && currentDtype === cacheKey) {
    self.postMessage({
      status: "device_ready",
      device: currentDevice,
      dtype: targetDtype,
      mobile: isMobileWebGPU || requestedMode === "wasm-mobile"
    });
    return ttsInstance;
  }

  const progressCb = (p) => {
    if (p.status === "progress" && p.total > 0) {
      const pct = Math.round((p.loaded / p.total) * 100);
      self.postMessage({
        status: "progress",
        data: { percent: pct, file: p.file || "weights", message: `Downloading ${p.file || "weights"}: ${pct}%` }
      });
    } else if (p.status === "done") {
      self.postMessage({
        status: "progress",
        data: { percent: 100, file: "weights", message: "Weights cached locally! Ready to speak." }
      });
    }
  };

  // 1. Attempt WebGPU hardware acceleration (Desktop Studio or Mobile Low-VRAM WebGPU)
  if (canUseWebGPU) {
    try {
      const modeLabel = isMobileWebGPU ? "Mobile WebGPU (Low-VRAM Mode ⚡)" : "WebGPU (FP32 Hardware Accelerated 🚀)";
      self.postMessage({
        status: "progress",
        data: { percent: 0, file: "Connecting...", message: `Initializing Kokoro on ${modeLabel}...` }
      });

      ttsInstance = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
        dtype: "fp32",
        device: "webgpu",
        progress_callback: progressCb
      });
      currentDevice = "webgpu";
      currentDtype = cacheKey;
      self.postMessage({ status: "device_ready", device: "webgpu", dtype: "fp32", mobile: isMobileWebGPU });
      return ttsInstance;
    } catch (gpuErr) {
      console.warn("WebGPU initialization failed, falling back to WASM CPU:", gpuErr);
      ttsInstance = null;
    }
  }

  // 2. Fallback to WebAssembly CPU with q8 (Rock-solid, crystal-clear, ~85 MB)
  self.postMessage({
    status: "progress",
    data: { percent: 0, file: "Connecting...", message: `Initializing Kokoro on WebAssembly (${requestedMode === "wasm-mobile" ? "Mobile Fast Mode 📱" : "Safe Mode 💻"})...` }
  });

  ttsInstance = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
    dtype: "q8",
    device: "wasm",
    progress_callback: progressCb
  });
  currentDevice = "wasm";
  currentDtype = `wasm-q8`;
  self.postMessage({ status: "device_ready", device: "wasm", dtype: "q8", mobile: requestedMode === "wasm-mobile" });
  return ttsInstance;
}

/**
 * Splits text into natural sentence/clause chunks under maxChars to prevent model truncation.
 * On WebGPU, keeping chunks under ~240 characters ensures each inference forward pass
 * executes in under ~500ms, completely avoiding GPU command queue saturation and preventing
 * macOS WindowServer (display compositor) freezes on Apple Silicon and shared-memory GPUs.
 */
function chunkText(text, maxChars = 240) {
  const rawParagraphs = text.split(/\n+/).map(p => p.trim()).filter(Boolean);
  const chunks = [];

  for (const para of rawParagraphs) {
    const sentences = para.match(/[^.!?]+[.!?]+(\s|$)|[^.!?]+$/g) || [para];
    let current = "";

    for (let sentence of sentences) {
      sentence = sentence.trim();
      if (!sentence) continue;

      if ((current + " " + sentence).trim().length <= maxChars) {
        current = current ? current + " " + sentence : sentence;
      } else {
        if (current) chunks.push(current);
        if (sentence.length > maxChars) {
          const clauses = sentence.match(/[^,;:—]+[,;:—]+(\s|$)|[^,;:—]+$/g) || [sentence];
          let sub = "";
          for (let clause of clauses) {
            clause = clause.trim();
            if ((sub + " " + clause).trim().length <= maxChars) {
              sub = sub ? sub + " " + clause : clause;
            } else {
              if (sub) chunks.push(sub);
              sub = clause;
            }
          }
          if (sub) chunks.push(sub);
          current = "";
        } else {
          current = sentence;
        }
      }
    }
    if (current) chunks.push(current);
  }

  return chunks.filter(c => c.trim().length > 0);
}

/**
 * Creates standard 16-bit PCM WAV Blob from raw Float32Array samples.
 */
function createWavBlob(float32Array, sampleRate = 24000) {
  const buffer = new ArrayBuffer(44 + float32Array.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + float32Array.length * 2, true);
  writeString(8, "WAVE");

  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);

  writeString(36, "data");
  view.setUint32(40, float32Array.length * 2, true);

  let offset = 44;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

let currentJobId = null;
let cancelRequested = false;

self.onmessage = async (e) => {
  const { type, text, voice, dtype, speed, id } = e.data;

  if (type === "cancel") {
    cancelRequested = true;
    return;
  }

  if (type === "preload") {
    try {
      await getOrInitTTS(dtype || "q4");
      self.postMessage({ status: "ready", id });
    } catch (err) {
      self.postMessage({ status: "error", id, message: err.message || String(err) });
    }
    return;
  }

  if (type === "generate") {
    currentJobId = id;
    cancelRequested = false;

    try {
      const tts = await getOrInitTTS(dtype || "q8");
      const startTime = performance.now();
      const speedValue = speed ? Math.max(0.5, Math.min(2.0, parseFloat(speed))) : 1.0;

      // Dynamic chunk sizing:
      // - Mobile WebGPU: 110 chars keeps transient GPU activation tensors small so mobile Safari/Chrome never exceed per-tab memory limits
      // - Desktop WebGPU: 220 chars (~1-2 sentences) keeps each forward pass around ~400ms, preventing GPU command queue saturation
      // - WASM CPU: 320 chars provides optimal SIMD batch throughput.
      const isMobileGpu = currentDevice === "webgpu" && dtype === "webgpu-mobile";
      const targetMaxChars = isMobileGpu ? 110 : (currentDevice === "webgpu" ? 220 : 320);
      const chunks = chunkText(text, targetMaxChars);
      if (chunks.length === 0) {
        throw new Error("Please enter text to synthesize.");
      }

      // Initial breather to allow browser GPU context and VRAM allocation to settle
      await new Promise((resolve) => setTimeout(resolve, isMobileGpu ? 90 : 50));

      const audioSegments = [];
      let sampleRate = 24000;
      const selectedVoice = voice || "af_heart";
      const isPortugueseVoice = Boolean(PORTUGUESE_VOICE_MAP[selectedVoice]);

      for (let i = 0; i < chunks.length; i++) {
        // Check if user requested cancellation
        if (cancelRequested || currentJobId !== id) {
          if (audioSegments.length > 0) {
            const totalSamples = audioSegments.reduce((acc, seg) => acc + seg.length, 0);
            const combinedBuffer = new Float32Array(totalSamples);
            let offset = 0;
            for (const seg of audioSegments) {
              combinedBuffer.set(seg, offset);
              offset += seg.length;
            }
            const partialBlob = createWavBlob(combinedBuffer, sampleRate);
            const partialDuration = (totalSamples / sampleRate).toFixed(1);
            self.postMessage({
              status: "cancelled",
              id,
              partialBlob,
              partialDuration,
              chunksProcessed: i,
              totalChunks: chunks.length,
              message: `Synthesis stopped. Preserved ${partialDuration}s audio (${i}/${chunks.length} parts).`
            });
          } else {
            self.postMessage({
              status: "cancelled",
              id,
              message: "Synthesis cancelled."
            });
          }
          return;
        }

        const chunk = chunks[i];
        const progressPct = Math.round(((i + 1) / chunks.length) * 100);

        // Calculate dynamic estimated remaining time based on observed speed
        let estRemainingMsg = "";
        if (i > 0) {
          const avgChunkTime = (performance.now() - startTime) / i;
          const remainingChunks = chunks.length - i;
          const estSec = Math.max(1, Math.round((avgChunkTime * remainingChunks) / 1000));
          estRemainingMsg = ` (~${estSec}s remaining)`;
        }

        self.postMessage({
          status: "synthesizing",
          id,
          current: i + 1,
          total: chunks.length,
          percent: progressPct,
          message: `Synthesizing part ${i + 1} of ${chunks.length}${estRemainingMsg}...`
        });

        const res = isPortugueseVoice
          ? await generatePortugueseChunk(tts, chunk, selectedVoice, speedValue)
          : await tts.generate(chunk, {
              voice: selectedVoice,
              speed: speedValue
            });

        if (res.sampling_rate) {
          sampleRate = res.sampling_rate;
        }

        if (res.audio && res.audio.length > 0) {
          audioSegments.push(res.audio);

          // Dispatch individual chunk immediately for real-time progressive streaming
          const chunkBlob = createWavBlob(res.audio, sampleRate);
          const chunkDuration = (res.audio.length / sampleRate).toFixed(1);
          self.postMessage({
            status: "chunk_ready",
            id,
            chunkIndex: i,
            totalChunks: chunks.length,
            chunkBlob,
            chunkDuration,
            percent: progressPct
          });

          // Insert natural breath pause between sentences
          if (i < chunks.length - 1) {
            const pauseSamples = new Float32Array(Math.round(sampleRate * 0.16));
            audioSegments.push(pauseSamples);
          }
        }

        // Cooperative GPU & CPU yield between chunks:
        // 160ms on Mobile WebGPU allows mobile OS memory manager to reclaim temporary tensor buffers,
        // 90ms on Desktop WebGPU allows macOS WindowServer / display compositor to render frames smoothly.
        if (i < chunks.length - 1) {
          const yieldMs = isMobileGpu ? 160 : (currentDevice === "webgpu" ? 90 : 30);
          await new Promise((resolve) => setTimeout(resolve, yieldMs));
        }
      }

      const totalSamples = audioSegments.reduce((acc, seg) => acc + seg.length, 0);
      if (audioSegments.length === 0 || totalSamples === 0) {
        throw new Error("Neural synthesis produced empty audio. Please check your text or try CPU Safe Mode.");
      }
      const combinedBuffer = new Float32Array(totalSamples);
      let offset = 0;
      for (const seg of audioSegments) {
        combinedBuffer.set(seg, offset);
        offset += seg.length;
      }

      const totalDurationSec = (totalSamples / sampleRate).toFixed(1);
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      const blob = createWavBlob(combinedBuffer, sampleRate);

      self.postMessage({
        status: "complete",
        id,
        blob,
        elapsed,
        totalDuration: totalDurationSec,
        chunksProcessed: chunks.length,
        message: `Generated ${totalDurationSec}s of audio in ${elapsed}s`
      });
    } catch (err) {
      console.error("Worker TTS error:", err);
      self.postMessage({
        status: "error",
        id,
        message: err.message || String(err)
      });
    }
  }
};
