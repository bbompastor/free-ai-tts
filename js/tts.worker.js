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

// Map multilingual voice IDs (pt-PT, pt-BR, es-ES, es-419, fr-FR, it-IT, hi-IN, ja-JP, zh-CN) to Kokoro v1.0 voice embedding files
const PORTUGUESE_VOICE_MAP = {
  // 🇵🇹 European Portuguese (Portugal)
  pt_dora: { bin: "pf_dora", lang: "pt-PT" },
  pt_alex: { bin: "pm_alex", lang: "pt-PT" },
  pt_santa: { bin: "pm_santa", lang: "pt-PT" },
  // 🇧🇷 Brazilian Portuguese (Brasil)
  ptbr_dora: { bin: "pf_dora", lang: "pt-BR" },
  ptbr_alex: { bin: "pm_alex", lang: "pt-BR" },
  ptbr_santa: { bin: "pm_santa", lang: "pt-BR" },
  pf_dora: { bin: "pf_dora", lang: "pt-BR" },
  pm_alex: { bin: "pm_alex", lang: "pt-BR" },
  pm_santa: { bin: "pm_santa", lang: "pt-BR" },
  // 🇪🇸 Spanish (Spain — Castilian)
  es_dora: { bin: "ef_dora", lang: "es-ES" },
  es_alex: { bin: "em_alex", lang: "es-ES" },
  es_santa: { bin: "em_santa", lang: "es-ES" },
  // 🇲🇽 Spanish (Latin America)
  esla_dora: { bin: "ef_dora", lang: "es-419" },
  esla_alex: { bin: "em_alex", lang: "es-419" },
  esla_santa: { bin: "em_santa", lang: "es-419" },
  ef_dora: { bin: "ef_dora", lang: "es-419" },
  em_alex: { bin: "em_alex", lang: "es-419" },
  em_santa: { bin: "em_santa", lang: "es-419" },
  // 🇫🇷 French (France)
  ff_siwis: { bin: "ff_siwis", lang: "fr-FR" },
  // 🇮🇹 Italian (Italy)
  if_sara: { bin: "if_sara", lang: "it-IT" },
  im_nicola: { bin: "im_nicola", lang: "it-IT" },
  // 🇮🇳 Hindi (India)
  hf_alpha: { bin: "hf_alpha", lang: "hi-IN" },
  hf_beta: { bin: "hf_beta", lang: "hi-IN" },
  hm_omega: { bin: "hm_omega", lang: "hi-IN" },
  hm_psi: { bin: "hm_psi", lang: "hi-IN" },
  // 🇯🇵 Japanese (Japan)
  jf_alpha: { bin: "jf_alpha", lang: "ja-JP" },
  jf_gongitsune: { bin: "jf_gongitsune", lang: "ja-JP" },
  jf_nezumi: { bin: "jf_nezumi", lang: "ja-JP" },
  jf_tebukuro: { bin: "jf_tebukuro", lang: "ja-JP" },
  jm_kumo: { bin: "jm_kumo", lang: "ja-JP" },
  // 🇨🇳 Mandarin Chinese (China)
  zf_xiaobei: { bin: "zf_xiaobei", lang: "zh-CN" },
  zf_xiaoni: { bin: "zf_xiaoni", lang: "zh-CN" },
  zf_xiaoxiao: { bin: "zf_xiaoxiao", lang: "zh-CN" },
  zf_xiaoyi: { bin: "zf_xiaoyi", lang: "zh-CN" },
  zm_yunjian: { bin: "zm_yunjian", lang: "zh-CN" },
  zm_yunxi: { bin: "zm_yunxi", lang: "zh-CN" },
  zm_yunxia: { bin: "zm_yunxia", lang: "zh-CN" },
  zm_yunyang: { bin: "zm_yunyang", lang: "zh-CN" }
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
 * Converts a Portuguese word into Brazilian Portuguese (pt-BR) IPA phonemes.
 * Implements standard Brazilian phonology:
 * - Affrication/palatalization of /t/ and /d/ before /i/ -> /ʧ/ and /ʤ/
 * - L-vocalization in syllable coda -> /w/ (e.g. "Brasil" -> bɾazˈiw, "artificial" -> aɾʧifisjˈaw)
 * - Alveolar coda sibilants (/s/, /z/ instead of European /ʃ/, /ʒ/)
 * - Unstressed final -e(s) raised to /i(s)/, internal unstressed 'e' kept as /e/
 */
function phonemizePortugueseBrazilWord(rawWord) {
  const w = rawWord.toLowerCase().trim();
  if (!w) return "";
  const brLexicon = {
    "de": "ʤi", "do": "du", "da": "dɐ", "dos": "dus", "das": "dɐs",
    "te": "ʧi", "ti": "ʧˈi", "me": "mi", "se": "si", "lhe": "ʎi", "lhes": "ʎis",
    "o": "u", "a": "a", "os": "us", "as": "as", "e": "i", "mas": "mas", "mais": "mˈajs",
    "em": "ẽj̃", "no": "nu", "na": "nɐ", "nos": "nus", "nas": "nɐs",
    "com": "kˈõ", "sem": "sˈẽj̃", "bem": "bˈẽj̃", "tem": "tˈẽj̃", "também": "tɐ̃bˈẽj̃",
    "você": "vosˈe", "vocês": "vosˈes", "ele": "ˈeli", "ela": "ˈɛlɐ", "eles": "ˈelis", "elas": "ˈɛlɐs",
    "este": "ˈesʧi", "esta": "ˈɛstɐ", "isto": "ˈistu", "esse": "ˈesi", "essa": "ˈɛsɐ", "isso": "ˈisu",
    "noite": "nˈojʧi", "tarde": "tˈaɾʤi", "hoje": "ˈoʒi", "onde": "ˈõʤi", "pode": "pˈɔʤi",
    "brasil": "bɾazˈiw", "português": "poɾtuɡˈes", "voz": "vˈɔs", "luz": "lˈus", "paz": "pˈas", "vez": "vˈes"
  };
  if (brLexicon[w]) return brLexicon[w];

  let ptIpa = phonemizePortugueseWord(w);
  // Convert European Portuguese IPA features into Brazilian Portuguese IPA:
  // 1. Word-final unstressed ɨ or ɨʃ -> i / is
  ptIpa = ptIpa.replace(/ɨʃ$/g, "is").replace(/ɨ$/g, "i");
  // 2. Internal unstressed ɨ -> e
  ptIpa = ptIpa.replace(/ɨ/g, "e");
  // 3. European -ɐj- diphthong -> -ej- and -ɐ̃j̃ -> -ẽj̃
  ptIpa = ptIpa.replace(/ɐj/g, "ej").replace(/ɐ̃j̃/g, "ẽj̃");
  // 4. Coda 'l' (at word end or before consonant) -> 'w'
  ptIpa = ptIpa.replace(/l(?=$|[pbtdkɡfvszʃʒmnɲʎɾʁ])/g, "w");
  // 5. Coda 'ʃ' (where source word had 's' or 'z', not 'ch'/'x') -> 's'
  if (!w.includes("ch") && !w.includes("x")) {
    ptIpa = ptIpa.replace(/ʃ$/g, "s").replace(/ʃ(?=[ptkf])/g, "s").replace(/ʒ(?=[bdɡvmnlr])/g, "z");
  }
  // 6. Palatalize /t/ and /d/ before /i/, /ˈi/, /ĩ/, /ˈĩ/, /j/ -> /ʧ/ and /ʤ/
  ptIpa = ptIpa
    .replace(/t(?=ˈ?[iĩj])/g, "ʧ")
    .replace(/d(?=ˈ?[iĩj])/g, "ʤ");

  return ptIpa;
}

const ES_UNITS = ["θˈeɾo", "ˈuno", "dˈos", "tɾˈes", "kwˈatɾo", "θˈinko", "sˈejs", "sjˈete", "ˈoʧo", "nwˈeβe"];
const ES_TEENS = ["djˈeθ", "ˈonθe", "dˈoθe", "tɾˈeθe", "katˈoɾθe", "kˈinθe", "djeθisˈejs", "djeθisjˈete", "djeθjˈoʧo", "djeθinwˈeβe"];
const ES_TENS = ["", "", "bˈejnte", "tɾˈejnta", "kwaɾˈenta", "θinkwˈenta", "sesˈenta", "setˈenta", "oʧˈenta", "nobˈenta"];

function numberToSpanishIPA(n, isSpain = true) {
  const fixSeseo = (s) => (isSpain ? s : s.replace(/θ/g, "s"));
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 10) return fixSeseo(ES_UNITS[n]);
  if (n < 20) return fixSeseo(ES_TEENS[n - 10]);
  if (n === 20) return fixSeseo("bˈejnte");
  if (n < 30) return fixSeseo(`bejnti${ES_UNITS[n - 20]}`);
  if (n < 100) {
    const t = Math.floor(n / 10);
    const u = n % 10;
    return fixSeseo(u === 0 ? ES_TENS[t] : `${ES_TENS[t]} i ${ES_UNITS[u]}`);
  }
  if (n === 100) return fixSeseo("θjˈen");
  if (n < 1000) {
    const h = Math.floor(n / 100);
    const rem = n % 100;
    const hMap = ["", "θjˈento", "dosθjˈentos", "tɾesθjˈentos", "kwatɾoθjˈentos", "kinjˈentos", "sejsθjˈentos", "seteθjˈentos", "oʧoθjˈentos", "nwebeθjˈentos"];
    return fixSeseo(rem === 0 ? hMap[h] : `${hMap[h]} ${numberToSpanishIPA(rem, true)}`);
  }
  return String(n);
}

/**
 * Rule-based Spanish (es-ES Castilian & es-419 Latin America) word-to-IPA converter.
 */
function phonemizeSpanishWord(rawWord, isSpain = true) {
  const w = rawWord.toLowerCase().trim();
  if (!w) return "";
  if (/^\d+$/.test(w)) return numberToSpanishIPA(parseInt(w, 10), isSpain);
  if (w.includes("-")) {
    return w.split("-").filter(Boolean).map(p => phonemizeSpanishWord(p, isSpain)).join(" ");
  }

  const esLexicon = {
    "y": "i", "o": "o", "u": "u", "el": "el", "la": "la", "los": "los", "las": "las",
    "un": "ˈun", "una": "ˈuna", "de": "de", "del": "del", "al": "al", "en": "en",
    "con": "kon", "por": "poɾ", "para": "pˈaɾa", "sin": "sin", "que": "ke", "se": "se",
    "no": "nˈo", "sí": "sˈi", "más": "mˈas", "muy": "mˈuj", "todo": "tˈoðo", "todos": "tˈoðos",
    "puretts": "pˈuɾe tˈe tˈe ˈese", "tts": "tˈe tˈe ˈese", "ia": "ˈi ˈa", "ai": "ˈaj"
  };
  if (esLexicon[w]) return esLexicon[w];

  const chars = Array.from(w);
  const len = chars.length;
  const isVowel = (c) => Boolean(c) && "aáeéiíoóuúü".includes(c);
  const hasAccent = (c) => Boolean(c) && "áéíóú".includes(c);
  const isStrongVowel = (c) => Boolean(c) && "aáeéoóíú".includes(c);
  const sibilantC = isSpain ? "θ" : "s";

  // Find syllable nuclei for Spanish stress placement
  const vowelIndices = [];
  for (let i = 0; i < len; i++) {
    const c = chars[i];
    const prev = i > 0 ? chars[i - 1] : "";
    const next = i + 1 < len ? chars[i + 1] : "";
    if (c === "u" && (prev === "q" || prev === "g") && (next === "e" || next === "é" || next === "i" || next === "í")) {
      continue;
    }
    if (isVowel(c) || (c === "y" && i === len - 1)) {
      vowelIndices.push(i);
    }
  }

  if (vowelIndices.length === 0) return w;

  const nuclei = [];
  let idx = 0;
  while (idx < vowelIndices.length) {
    const vIdx = vowelIndices[idx];
    const c = chars[vIdx];
    const nextVIdx = idx + 1 < vowelIndices.length ? vowelIndices[idx + 1] : -1;
    if (nextVIdx === vIdx + 1) {
      const nextC = chars[nextVIdx];
      // Two weak vowels (i/u) or one strong + one unaccented weak form a single syllable diphthong
      if (!(isStrongVowel(c) && isStrongVowel(nextC))) {
        nuclei.push({ start: vIdx, end: nextVIdx });
        idx += 2;
        continue;
      }
    }
    nuclei.push({ start: vIdx, end: vIdx });
    idx++;
  }

  let stressedNucleusIdx = -1;
  for (let n = 0; n < nuclei.length; n++) {
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
      const lastChar = chars[len - 1];
      if ("aeiouns".includes(lastChar)) {
        stressedNucleusIdx = nuclei.length - 2;
      } else {
        stressedNucleusIdx = nuclei.length - 1;
      }
    }
  }

  const stressedPos = nuclei[stressedNucleusIdx].start;
  let ipa = "";
  let i = 0;

  while (i < len) {
    const c = chars[i];
    const prev = i > 0 ? chars[i - 1] : "";
    const next = i + 1 < len ? chars[i + 1] : "";
    const next2 = i + 2 < len ? chars[i + 2] : "";
    const stress = i === stressedPos ? "ˈ" : "";

    if (c === "c" && next === "h") { ipa += "ʧ"; i += 2; continue; }
    if (c === "l" && next === "l") { ipa += "ʎ"; i += 2; continue; }
    if (c === "r" && next === "r") { ipa += "r"; i += 2; continue; }
    if (c === "q" && next === "u" && "eéíi".includes(next2)) { ipa += "k"; i += 2; continue; }
    if (c === "g" && next === "u" && "eéíi".includes(next2)) { ipa += "ɡ"; i += 2; continue; }
    if (c === "g" && next === "ü") { ipa += "ɡw"; i += 2; continue; }

    // Diphthongs with weak i/u + vowel
    if ((c === "i" || c === "u" || c === "ü") && !hasAccent(c) && isVowel(next)) {
      const glide = c === "i" ? "j" : "w";
      const nextStress = (i === stressedPos || i + 1 === stressedPos) ? "ˈ" : "";
      const vMap = { "a": "a", "á": "a", "e": "e", "é": "e", "i": "i", "í": "i", "o": "o", "ó": "o", "u": "u", "ú": "u" };
      ipa += `${nextStress}${glide}${vMap[next] || next}`;
      i += 2;
      continue;
    }

    // Falling diphthongs (strong vowel + unaccented i/u/y)
    if ("aáeéoó".includes(c) && (next === "i" || next === "u" || (next === "y" && i + 1 === len - 1))) {
      const vMap = { "a": "a", "á": "a", "e": "e", "é": "e", "o": "o", "ó": "o" };
      const glide = next === "u" ? "w" : "j";
      ipa += `${stress}${vMap[c]}${glide}`;
      i += 2;
      continue;
    }

    if (isVowel(c)) {
      const vMap = { "a": "a", "á": "a", "e": "e", "é": "e", "i": "i", "í": "i", "o": "o", "ó": "o", "u": "u", "ú": "u", "ü": "w" };
      ipa += `${stress}${vMap[c]}`;
      i++;
      continue;
    }

    switch (c) {
      case "b":
      case "v":
        ipa += (i === 0 || prev === "m" || prev === "n") ? "b" : "β";
        break;
      case "c":
        ipa += "eéíi".includes(next) ? sibilantC : "k";
        break;
      case "d":
        ipa += (i === 0 || prev === "n" || prev === "l") ? "d" : "ð";
        break;
      case "f":
      case "k":
      case "l":
      case "m":
      case "p":
      case "s":
      case "t":
      case "w":
        ipa += c;
        break;
      case "g":
        ipa += "eéíi".includes(next) ? "x" : "ɡ";
        break;
      case "h":
        break;
      case "j":
        ipa += "x";
        break;
      case "n":
        ipa += (next && "kgx".includes(next)) ? "ŋ" : "n";
        break;
      case "ñ":
        ipa += "ɲ";
        break;
      case "r":
        ipa += (i === 0 || prev === "n" || prev === "l" || prev === "s") ? "r" : "ɾ";
        break;
      case "x":
        ipa += "ks";
        break;
      case "y":
        ipa += (i === len - 1) ? `${stress}i` : "ʝ";
        break;
      case "z":
        ipa += sibilantC;
        break;
      default:
        break;
    }
    i++;
  }
  return ipa;
}

/**
 * Rule-based Italian (it-IT) word-to-IPA converter.
 */
function phonemizeItalianWord(rawWord) {
  const w = rawWord.toLowerCase().trim();
  if (!w) return "";
  if (w.includes("-")) {
    return w.split("-").filter(Boolean).map(p => phonemizeItalianWord(p)).join(" ");
  }

  const itLexicon = {
    "il": "il", "lo": "lo", "la": "la", "i": "i", "gli": "ʎi", "le": "le",
    "un": "un", "uno": "ˈuno", "una": "ˈuna", "di": "di", "a": "a", "da": "da",
    "in": "in", "con": "kon", "su": "su", "per": "peɾ", "tra": "tɾa", "fra": "fɾa",
    "e": "e", "ed": "ed", "o": "o", "ma": "ma", "che": "ke", "chi": "ki", "non": "non",
    "è": "ˈɛ", "sono": "sˈono", "ha": "ˈa", "hanno": "ˈanno", "più": "pjˈu", "già": "ʤˈa",
    "può": "pwˈɔ", "ciò": "ʧˈɔ", "perché": "peɾkˈe", "così": "kozˈi", "città": "ʧittˈa",
    "tutto": "tˈutto", "tutti": "tˈutti", "nel": "nel", "nella": "nˈella", "sul": "sul", "del": "del",
    "suo": "sˈuo", "sua": "sˈua", "suoi": "swˈɔj", "mio": "mˈio", "mia": "mˈia",
    "studio": "stˈudjo", "voce": "vˈoʧe", "intelligenza": "intelliʤˈɛntsa", "artificiale": "aɾtifiʧˈale",
    "direttamente": "diɾettamˈente", "browser": "bɾˈawzeɾ", "elaborazione": "elaboɾatsjˈone",
    "avviene": "avvjˈɛne", "localmente": "lokalmˈente", "dispositivo": "dispozitˈivo",
    "veloce": "velˈoʧe", "privato": "pɾivˈato", "gratuito": "ɡɾatˈuito", "cento": "ʧˈɛnto",
    "puretts": "pˈuɾe tˈi tˈi ˈɛsse", "tts": "tˈi tˈi ˈɛsse", "ia": "ˈi ˈa", "ai": "ˈaj"
  };
  if (itLexicon[w]) return itLexicon[w];
  if (w === "100") return "ʧˈɛnto";

  const chars = Array.from(w);
  const len = chars.length;
  const isVowel = (c) => Boolean(c) && "aàeèéiìoòóuù".includes(c);
  const hasAccent = (c) => Boolean(c) && "àèéìòóù".includes(c);

  const vowelPositions = [];
  for (let i = 0; i < len; i++) {
    const c = chars[i];
    const prev = i > 0 ? chars[i - 1] : "";
    const next = i + 1 < len ? chars[i + 1] : "";
    // Skip orthographic 'i' in ci/gi/sci + a/o/u/e
    if (c === "i" && (prev === "c" || prev === "g" || prev === "l") && isVowel(next)) {
      continue;
    }
    if (isVowel(c)) vowelPositions.push(i);
  }
  if (vowelPositions.length === 0) return w;

  let stressedPos = vowelPositions[Math.max(0, vowelPositions.length - 2)];
  for (const pos of vowelPositions) {
    if (hasAccent(chars[pos])) {
      stressedPos = pos;
      break;
    }
  }

  let ipa = "";
  let i = 0;
  while (i < len) {
    const c = chars[i];
    const prev = i > 0 ? chars[i - 1] : "";
    const next = i + 1 < len ? chars[i + 1] : "";
    const next2 = i + 2 < len ? chars[i + 2] : "";
    const stress = i === stressedPos ? "ˈ" : "";

    if (c === "g" && next === "l" && next2 === "i") {
      if (i + 3 < len && isVowel(chars[i + 3])) {
        ipa += "ʎ";
        i += 3;
      } else {
        ipa += `ʎ${i + 2 === stressedPos ? "ˈ" : ""}i`;
        i += 3;
      }
      continue;
    }
    if (c === "g" && next === "n") { ipa += "ɲ"; i += 2; continue; }
    if (c === "s" && next === "c" && "eèéiì".includes(next2)) {
      if (next2 === "i" && i + 3 < len && isVowel(chars[i + 3])) {
        ipa += "ʃ";
        i += 3;
      } else {
        ipa += "ʃ";
        i += 2;
      }
      continue;
    }
    if (c === "c" && next === "h") { ipa += "k"; i += 2; continue; }
    if (c === "g" && next === "h") { ipa += "ɡ"; i += 2; continue; }
    if (c === "q" && next === "u") { ipa += "kw"; i += 2; continue; }
    if (c === "c" && next === "i" && isVowel(next2)) { ipa += "ʧ"; i += 2; continue; }
    if (c === "g" && next === "i" && isVowel(next2)) { ipa += "ʤ"; i += 2; continue; }

    if ((c === "i" || c === "u") && !hasAccent(c) && isVowel(next) && i !== stressedPos) {
      ipa += c === "i" ? "j" : "w";
      i++;
      continue;
    }

    if (isVowel(c)) {
      const vMap = { "a": "a", "à": "a", "e": "e", "é": "e", "è": "ɛ", "i": "i", "ì": "i", "o": "o", "ó": "o", "ò": "ɔ", "u": "u", "ù": "u" };
      ipa += `${stress}${vMap[c]}`;
      i++;
      continue;
    }

    switch (c) {
      case "c":
        ipa += "eèéiì".includes(next) ? "ʧ" : "k";
        break;
      case "g":
        ipa += "eèéiì".includes(next) ? "ʤ" : "ɡ";
        break;
      case "h":
        break;
      case "r":
        ipa += "ɾ";
        break;
      case "s":
        ipa += (isVowel(prev) && isVowel(next)) ? "z" : "s";
        break;
      case "z":
        ipa += "ts";
        break;
      default:
        ipa += c;
        break;
    }
    i++;
  }
  return ipa;
}

/**
 * Rule-based & lexicon French (fr-FR) word-to-IPA converter.
 */
function phonemizeFrenchWord(rawWord) {
  const w = rawWord.toLowerCase().trim().replace(/’/g, "'");
  if (!w) return "";
  if (w.includes("'")) {
    const parts = w.split("'");
    const prefixMap = { "l": "l", "d": "d", "j": "ʒ", "m": "m", "t": "t", "s": "s", "c": "s", "n": "n", "qu": "k" };
    const pref = prefixMap[parts[0]] || phonemizeFrenchWord(parts[0]);
    const rest = phonemizeFrenchWord(parts.slice(1).join(""));
    return `${pref}${rest}`;
  }
  if (w.includes("-")) {
    return w.split("-").filter(Boolean).map(p => phonemizeFrenchWord(p)).join(" ");
  }

  const frLexicon = {
    "bonjour": "bɔ̃ʒˈuʁ", "bienvenue": "bjɛ̃vənˈy", "sur": "syʁ", "votre": "vˈɔtʁ",
    "studio": "stydjˈo", "de": "də", "voix": "vwˈa", "avec": "avˈɛk",
    "intelligence": "ɛ̃tɛliʒˈɑ̃s", "artificielle": "aʁtifisjˈɛl", "entièrement": "ɑ̃tjɛʁmˈɑ̃",
    "dans": "dɑ̃", "le": "lə", "la": "la", "les": "le", "un": "ɛ̃", "une": "ˈyn", "des": "de",
    "du": "dy", "au": "o", "aux": "o", "en": "ɑ̃", "et": "e", "ou": "u", "où": "u",
    "est": "ˈɛ", "sont": "sˈɔ̃", "a": "a", "ont": "ˈɔ̃", "pour": "puʁ", "par": "paʁ",
    "sans": "sɑ̃", "plus": "ply", "tout": "tˈu", "tous": "tˈus", "toute": "tˈut",
    "navigateur": "naviɡatˈœʁ", "traitement": "tʁɛtmˈɑ̃", "s'effectue": "sefɛktˈy",
    "effectue": "efɛktˈy", "localement": "lokalmˈɑ̃", "appareil": "apaʁˈɛj",
    "rapide": "ʁapˈid", "privé": "pʁivˈe", "gratuit": "ɡʁatɥˈi", "cent": "sˈɑ̃", "100": "sˈɑ̃",
    "puretts": "pˈyʁ tˈe tˈe ˈɛs", "tts": "tˈe tˈe ˈɛs", "ia": "ˈi ˈa", "ai": "ˈɛj"
  };
  if (frLexicon[w]) return frLexicon[w];

  let s = w;
  // Strip silent final consonants (-e, -es, -s, -t, -d, -p, -x, -z) in polysyllabic words
  s = s
    .replace(/tion$/g, "sjˈɔ̃")
    .replace(/ment$/g, "mˈɑ̃")
    .replace(/eux$/g, "ˈø")
    .replace(/euse$/g, "ˈøz")
    .replace(/eur$/g, "ˈœʁ")
    .replace(/er$/g, "ˈe")
    .replace(/ez$/g, "ˈe")
    .replace(/es?$/g, "")
    .replace(/[stdpxz]$/g, "");

  s = s
    .replace(/eau/g, "o")
    .replace(/au/g, "o")
    .replace(/ou|où|oû/g, "u")
    .replace(/oi/g, "wa")
    .replace(/eu|œu/g, "ø")
    .replace(/ai|ei|è|ê|ë/g, "ɛ")
    .replace(/é/g, "e")
    .replace(/(?:an|am|en|em)(?![aàâeéèêiîoôuûy]|n|m)/g, "ɑ̃")
    .replace(/(?:in|im|ain|aim|ein)(?![aàâeéèêiîoôuûy]|n|m)/g, "ɛ̃")
    .replace(/(?:on|om)(?![aàâeéèêiîoôuûy]|n|m)/g, "ɔ̃")
    .replace(/ch/g, "ʃ")
    .replace(/gn/g, "ɲ")
    .replace(/ph/g, "f")
    .replace(/qu/g, "k")
    .replace(/gu(?=[eéèêiîy])/g, "ɡ")
    .replace(/c(?=[eéèêiîy])/g, "s")
    .replace(/ç/g, "s")
    .replace(/c/g, "k")
    .replace(/g(?=[eéèêiîy])/g, "ʒ")
    .replace(/g/g, "ɡ")
    .replace(/j/g, "ʒ")
    .replace(/r+/g, "ʁ")
    .replace(/u/g, "y")
    .replace(/h/g, "");

  if (!s.includes("ˈ")) {
    // Place French primary stress on final vowel
    s = s.replace(/([aɑeɛøœioɔuy])([^aɑeɛøœioɔuy]*)$/, "ˈ$1$2");
  }
  return s;
}

/**
 * Hindi (hi-IN) Devanagari & Hinglish to IPA G2P engine.
 * Implements Devanagari abugida consonant + matra + virama + nukta rules and word-final schwa deletion.
 */
function phonemizeHindiWord(rawWord) {
  const w = rawWord.trim();
  if (!w) return "";
  const lower = w.toLowerCase();
  const hiLexicon = {
    "puretts": "pjˈʊəɾ tˈiː tˈiː ˈɛs", "tts": "tˈiː tˈiː ˈɛs", "ai": "ˈeː ˈaːiː",
    "100": "sˈɔː", "१००": "sˈɔː", "namaste": "nəmˈəsteː", "bharat": "bʱˈaːɾət", "hindi": "ɦˈɪndiː"
  };
  if (hiLexicon[lower]) return hiLexicon[lower];

  // If word is Latin/Romanized Hinglish, apply simple Roman-to-Hindi IPA
  if (/^[a-z0-9-]+$/i.test(lower)) {
    return lower
      .replace(/sh/g, "ʃ").replace(/ch/g, "ʧ").replace(/kh/g, "kʰ").replace(/gh/g, "ɡʱ")
      .replace(/th/g, "tʰ").replace(/dh/g, "dʱ").replace(/ph/g, "pʰ").replace(/bh/g, "bʱ")
      .replace(/aa/g, "aː").replace(/ee/g, "iː").replace(/oo/g, "uː")
      .replace(/j/g, "ʤ").replace(/v|w/g, "ʋ").replace(/r/g, "ɾ").replace(/h/g, "ɦ").replace(/g/g, "ɡ");
  }

  // Devanagari abugida parser
  const indepVowels = {
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː", "उ": "ʊ", "ऊ": "uː",
    "ऋ": "ɾɪ", "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː", "ऑ": "ɔ"
  };
  const matras = {
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ", "ू": "uː",
    "ृ": "ɾɪ", "े": "eː", "ै": "ɛː", "ो": "oː", "ौ": "ɔː", "ॉ": "ɔ"
  };
  const consonants = {
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʱ", "ङ": "ŋ",
    "च": "ʧ", "छ": "ʧʰ", "ज": "ʤ", "झ": "ʤʱ", "ञ": "ɲ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʱ", "ण": "ɳ",
    "त": "t", "थ": "tʰ", "द": "d", "ध": "dʱ", "न": "n",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʱ", "म": "m",
    "य": "j", "र": "ɾ", "ल": "l", "व": "ʋ",
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "ɦ",
    "क़": "k", "ख़": "x", "ग़": "ɣ", "ज़": "z", "ड़": "ɽ", "ढ़": "ɽʱ", "फ़": "f"
  };
  const nuktaMap = {
    "क": "k", "ख": "x", "ग": "ɣ", "ज": "z", "ड": "ɽ", "ढ": "ɽʱ", "फ": "f"
  };

  const chars = Array.from(w);
  const len = chars.length;
  let ipa = "";
  let i = 0;

  while (i < len) {
    const c = chars[i];
    const next = i + 1 < len ? chars[i + 1] : "";

    if (indepVowels[c]) {
      ipa += indepVowels[c];
      i++;
      continue;
    }

    if (consonants[c]) {
      let consIpa = consonants[c];
      if (next === "़" && nuktaMap[c]) {
        consIpa = nuktaMap[c];
        i++;
      }
      const afterCons = i + 1 < len ? chars[i + 1] : "";
      if (afterCons === "्") {
        // Virama suppresses inherent schwa
        ipa += consIpa;
        i += 2;
        continue;
      } else if (matras[afterCons]) {
        ipa += consIpa + matras[afterCons];
        i += 2;
        continue;
      } else {
        // Inherent schwa (ə), deleted at word-end in spoken Hindi unless monosyllabic
        const isWordFinal = (i === len - 1) || (i === len - 2 && (afterCons === "ं" || afterCons === "ँ"));
        if (isWordFinal && len > 1) {
          ipa += consIpa;
        } else {
          ipa += consIpa + "ə";
        }
        i++;
        continue;
      }
    }

    if (matras[c]) {
      ipa += matras[c];
    } else if (c === "ं" || c === "ँ") {
      // Anusvara before stop or word-end -> nasalization tilde or nasal consonant
      if (next && "कखगघ".includes(next)) ipa += "ŋ";
      else if (next && "चछजझ".includes(next)) ipa += "ɲ";
      else if (next && "टठडढण".includes(next)) ipa += "ɳ";
      else if (next && "तथदधन".includes(next)) ipa += "n";
      else if (next && "पफबभम".includes(next)) ipa += "m";
      else ipa += "\u0303";
    } else if (c === "ः") {
      ipa += "ɦ";
    }
    i++;
  }

  // Place primary stress on the first long vowel or first vowel
  if (ipa && !ipa.includes("ˈ")) {
    if (/[aɪiʊueɛoɔ]ː/.test(ipa)) {
      ipa = ipa.replace(/([aɪiʊueɛoɔ]ː)/, "ˈ$1");
    } else {
      ipa = ipa.replace(/([əaɪiʊueɛoɔ])/, "ˈ$1");
    }
  }
  return ipa;
}

/**
 * Japanese (ja-JP) Kanji/Hiragana/Katakana/Romaji to IPA G2P engine for Kokoro-82M.
 */
const JA_KANJI_MAP = [
  ["こんにちは", "こんにちわ"], ["こんばんは", "こんばんわ"], ["これは", "これわ"], ["へようこそ", "えよーこそ"],
  ["人工知能", "じんこーちのー"], ["音声合成", "おんせいごーせい"], ["音声", "おんせい"],
  ["変換", "へんかん"], ["処理", "しょり"], ["完全", "かんぜん"], ["無料", "むりょー"],
  ["高速", "こーそく"], ["安全", "あんぜん"], ["直接", "ちょくせつ"], ["動作", "どーさ"],
  ["無制限", "むせいげん"], ["登録", "とーろく"], ["不要", "ふよー"], ["日本語", "にほんご"],
  ["日本", "にほん"], ["東京", "とーきょー"], ["世界", "せかい"], ["今日", "きょー"],
  ["明日", "あした"], ["昨日", "きのー"], ["私", "わたし"], ["僕", "ぼく"],
  ["彼", "かれ"], ["彼女", "かのじょ"], ["先生", "せんせい"], ["学生", "がくせい"],
  ["言葉", "ことば"], ["文章", "ぶんしょー"], ["文字", "もじ"], ["読", "よ"],
  ["聞", "き"], ["話", "はな"], ["声", "こえ"], ["音", "おと"], ["新", "あたら"],
  ["時間", "じかん"], ["今", "いま"], ["人", "ひと"], ["方", "かた"], ["本", "ほん"],
  ["一", "いち"], ["二", "に"], ["三", "さん"], ["四", "よん"], ["五", "ご"],
  ["六", "ろく"], ["七", "なな"], ["八", "はち"], ["九", "きゅー"], ["十", "じゅー"],
  ["百", "ひゃく"], ["千", "せん"], ["万", "まん"], ["100%", "ひゃくパーセント"],
  ["PureTTS", "ピュアティーティーエス"], ["TTS", "ティーティーエス"], ["AI", "エーアイ"]
];

const JA_DIGRAPHS = {
  "きゃ": "kʲa", "きゅ": "kʲɯ", "きょ": "kʲo",
  "しゃ": "ɕa", "しゅ": "ɕɯ", "しょ": "ɕo",
  "ちゃ": "ʧa", "ちゅ": "ʧɯ", "ちょ": "ʧo",
  "にゃ": "ɲa", "にゅ": "ɲɯ", "にょ": "ɲo",
  "ひゃ": "ça", "ひゅ": "çɯ", "ひょ": "ço",
  "みゃ": "mʲa", "みゅ": "mʲɯ", "みょ": "mʲo",
  "りゃ": "ɾʲa", "りゅ": "ɾʲɯ", "りょ": "ɾʲo",
  "ぎゃ": "ɡʲa", "ぎゅ": "ɡʲɯ", "ぎょ": "ɡʲo",
  "じゃ": "ʤa", "じゅ": "ʤɯ", "じょ": "ʤo",
  "びゃ": "bʲa", "びゅ": "bʲɯ", "びょ": "bʲo",
  "ぴゃ": "pʲa", "ぴゅ": "pʲɯ", "ぴょ": "pʲo",
  "ふぁ": "ɸa", "ふぃ": "ɸi", "ふぇ": "ɸe", "ふぉ": "ɸo",
  "てぃ": "ti", "でぃ": "di", "うぃ": "wi", "うぇ": "we", "うぉ": "wo"
};

const JA_MONOGRAPHS = {
  "あ": "a", "い": "i", "う": "ɯ", "え": "e", "お": "o",
  "ぁ": "a", "ぃ": "i", "ぅ": "ɯ", "ぇ": "e", "ぉ": "o",
  "か": "ka", "き": "kʲi", "く": "kɯ", "け": "ke", "こ": "ko",
  "さ": "sa", "し": "ɕi", "す": "sɯ", "せ": "se", "そ": "so",
  "た": "ta", "ち": "ʧi", "つ": "tsɯ", "て": "te", "と": "to",
  "な": "na", "に": "ɲi", "ぬ": "nɯ", "ね": "ne", "の": "no",
  "は": "ha", "ひ": "çi", "ふ": "ɸɯ", "へ": "he", "ほ": "ho",
  "ま": "ma", "み": "mʲi", "む": "mɯ", "め": "me", "も": "mo",
  "や": "ja", "ゆ": "jɯ", "よ": "jo", "ゃ": "ja", "ゅ": "jɯ", "ょ": "jo",
  "ら": "ɾa", "り": "ɾʲi", "る": "ɾɯ", "れ": "ɾe", "ろ": "ɾo",
  "わ": "wa", "を": "o", "ん": "ɴ",
  "が": "ɡa", "ぎ": "ɡʲi", "ぐ": "ɡɯ", "げ": "ɡe", "ご": "ɡo",
  "ざ": "dza", "じ": "ʤi", "ず": "dzɯ", "ぜ": "dze", "ぞ": "dzo",
  "だ": "da", "ぢ": "ʤi", "づ": "dzɯ", "で": "de", "ど": "do",
  "ば": "ba", "び": "bʲi", "ぶ": "bɯ", "べ": "be", "ぼ": "bo",
  "ぱ": "pa", "ぴ": "pʲi", "ぷ": "pɯ", "ぺ": "pe", "ぽ": "po",
  "ゔ": "vɯ", "ー": "ː"
};

function phonemizeJapaneseText(text) {
  let s = text
    .replace(/。/g, ". ")
    .replace(/、/g, ", ")
    .replace(/！/g, "! ")
    .replace(/？/g, "? ")
    .replace(/「|」|『|』/g, " ");

  for (const [kanji, reading] of JA_KANJI_MAP) {
    s = s.split(kanji).join(reading);
  }

  // Convert Katakana (ァ-ヶ) to Hiragana (ぁ-ゖ)
  let hira = "";
  for (const ch of s) {
    const code = ch.charCodeAt(0);
    if (code >= 0x30A1 && code <= 0x30F6) {
      hira += String.fromCharCode(code - 0x60);
    } else {
      hira += ch;
    }
  }

  const chars = Array.from(hira);
  let out = "";
  let i = 0;
  while (i < chars.length) {
    const c = chars[i];
    const next = i + 1 < chars.length ? chars[i + 1] : "";
    const pair = c + next;

    if (JA_DIGRAPHS[pair]) {
      out += JA_DIGRAPHS[pair];
      i += 2;
      continue;
    }
    if (c === "っ") {
      // Geminate next consonant
      const nextPh = JA_DIGRAPHS[next + (chars[i + 2] || "")] || JA_MONOGRAPHS[next] || "";
      if (nextPh) {
        out += nextPh[0];
      }
      i++;
      continue;
    }
    if (JA_MONOGRAPHS[c]) {
      // Particle は before space/punctuation or after noun phrases is often /wa/, keep natural
      out += JA_MONOGRAPHS[c];
      i++;
      continue;
    }
    if (/[,;:.!?—…\s]/.test(c)) {
      out += c;
      i++;
      continue;
    }
    if (/[A-Za-z]/.test(c)) {
      out += c.toLowerCase();
      i++;
      continue;
    }
    i++;
  }

  return out.replace(/\s+/g, " ").trim().normalize("NFD");
}

/**
 * Mandarin Chinese (zh-CN) Hanzi & Pinyin to IPA G2P engine for Kokoro-82M.
 * Lazily imports `pinyin-pro` in the worker when a Chinese voice is selected, with built-in fallback.
 */
let pinyinProModule = null;
async function getPinyinArray(text) {
  if (!pinyinProModule) {
    try {
      pinyinProModule = await import("https://esm.sh/pinyin-pro@3.18.2");
    } catch (e) {
      console.warn("Could not load pinyin-pro from CDN, using fallback dictionary:", e);
    }
  }
  if (pinyinProModule && typeof pinyinProModule.pinyin === "function") {
    return pinyinProModule.pinyin(text, { toneType: "num", type: "array", nonZh: "consecutive" });
  }
  return [text];
}

const ZH_FALLBACK_CHAR_MAP = {
  "你": "ni3", "好": "hao3", "欢": "huan1", "迎": "ying2", "使": "shi3", "用": "yong4",
  "这": "zhe4", "是": "shi4", "一": "yi1", "个": "ge4", "完": "wan2", "全": "quan2",
  "免": "mian3", "费": "fei4", "的": "de5", "人": "ren2", "工": "gong1", "智": "zhi4",
  "能": "neng2", "语": "yu3", "音": "yin1", "合": "he2", "成": "cheng2", "工": "gong1",
  "具": "ju4", "所": "suo3", "有": "you3", "处": "chu3", "理": "li3", "都": "dou1",
  "在": "zai4", "您": "nin2", "浏": "liu2", "览": "lan3", "器": "qi4", "中": "zhong1",
  "本": "ben3", "地": "di4", "运": "yun4", "行": "xing2", "快": "kuai4", "速": "su4",
  "私": "si1", "密": "mi4", "且": "qie3", "百": "bai3", "分": "fen1", "之": "zhi1"
};

const ZH_TONE_ARROWS = { "1": "→", "2": "↗", "3": "↓", "4": "↘", "5": "", "0": "" };

function pinyinSyllableToIPA(rawSyl) {
  const s = rawSyl.toLowerCase().trim();
  if (!s) return "";
  const m = s.match(/^([a-züv]+)([1-5])?$/);
  if (!m) return s;

  let base = m[1].replace(/v/g, "ü");
  const tone = m[2] || "5";
  const arrow = ZH_TONE_ARROWS[tone] || "";

  // Handle zero-initial w- and y- spellings in Pinyin
  if (base.startsWith("yu")) base = "ü" + base.slice(2);
  else if (base.startsWith("yi")) base = "i" + base.slice(2);
  else if (base.startsWith("y")) base = "i" + base.slice(1);
  else if (base.startsWith("wu")) base = "u" + base.slice(2);
  else if (base.startsWith("w")) base = "u" + base.slice(1);

  const initials = [
    ["zh", "ʈʂ"], ["ch", "ʈʂʰ"], ["sh", "ʂ"],
    ["b", "p"], ["p", "pʰ"], ["m", "m"], ["f", "f"],
    ["d", "t"], ["t", "tʰ"], ["n", "n"], ["l", "l"],
    ["g", "k"], ["k", "kʰ"], ["h", "x"],
    ["j", "tɕ"], ["q", "tɕʰ"], ["x", "ɕ"],
    ["r", "ɻ"], ["z", "ts"], ["c", "tsʰ"], ["s", "s"]
  ];

  let initIpa = "";
  let initRaw = "";
  for (const [k, v] of initials) {
    if (base.startsWith(k)) {
      initIpa = v;
      initRaw = k;
      base = base.slice(k.length);
      break;
    }
  }

  // After j, q, x, 'u' is always /y/ (ü)
  if ((initRaw === "j" || initRaw === "q" || initRaw === "x") && base.startsWith("u")) {
    base = "ü" + base.slice(1);
  }

  const finals = {
    "i": ["zh", "ch", "sh", "r", "z", "c", "s"].includes(initRaw) ? "ɨ" : "i",
    "u": "u", "ü": "y",
    "a": "a", "ia": "ja", "ua": "wa",
    "o": "wo", "uo": "wo",
    "e": "ɤ", "ie": "jɛ", "üe": "ɥɛ", "ue": "ɥɛ",
    "er": "ɚ",
    "ai": "aj", "uai": "waj",
    "ei": "ej", "ui": "wej", "uei": "wej",
    "ao": "aw", "iao": "jaw",
    "ou": "ow", "iu": "jow", "iou": "jow",
    "an": "an", "ian": "jɛn", "uan": "wan", "üan": "ɥɛn",
    "en": "ən", "in": "in", "un": "wən", "uen": "wən", "ün": "yn",
    "ang": "ɑŋ", "iang": "jɑŋ", "uang": "wɑŋ",
    "eng": "əŋ", "ing": "iŋ", "ueng": "wəŋ", "ong": "ʊŋ", "iong": "jʊŋ"
  };

  const finalIpa = finals[base] || base;
  return `${initIpa}${finalIpa}${arrow}`;
}

async function phonemizeChineseText(text) {
  const normalized = text
    .replace(/100%/g, "百分之百")
    .replace(/PureTTS/gi, " pˈjʊɹ tˈiː tˈiː ˈɛs ")
    .replace(/。/g, ". ")
    .replace(/，|、/g, ", ")
    .replace(/！/g, "! ")
    .replace(/？/g, "? ")
    .replace(/；/g, "; ")
    .replace(/：/g, ": ")
    .replace(/——/g, "— ");

  // Replace known characters if pinyin-pro fails to load offline
  const arr = await getPinyinArray(normalized);
  const out = [];

  for (const item of arr) {
    const trimmed = item.trim();
    if (!trimmed) continue;
    if (/^[,;:.!?—…"«»“”]+$/.test(trimmed)) {
      if (out.length > 0) out[out.length - 1] += trimmed;
      continue;
    }
    if (/^[a-züv]+[1-5]$/i.test(trimmed)) {
      out.push(pinyinSyllableToIPA(trimmed));
    } else if (/[\u4e00-\u9fff]/.test(trimmed)) {
      for (const ch of trimmed) {
        if (ZH_FALLBACK_CHAR_MAP[ch]) {
          out.push(pinyinSyllableToIPA(ZH_FALLBACK_CHAR_MAP[ch]));
        }
      }
    } else {
      out.push(trimmed);
    }
  }
  return out.join(" ").replace(/\s+/g, " ").trim().normalize("NFD");
}

/**
 * Phonemizes text into target language IPA phonemes (pt-PT, pt-BR, es-ES, es-419, fr-FR, it-IT, hi-IN, ja-JP, zh-CN)
 * while preserving punctuation so Kokoro-82M maintains natural prosody and clause pauses.
 */
async function phonemizePortugueseKeepPunctuation(text, lang = "pt-PT") {
  if (lang === "ja-JP") {
    return phonemizeJapaneseText(text);
  }
  if (lang === "zh-CN") {
    return await phonemizeChineseText(text);
  }
  if (lang === "hi-IN") {
    const hiNorm = text
      .replace(/।/g, ".")
      .replace(/(\d+)\s*%/g, "$1 प्रतिशत")
      .replace(/[०-९]/g, d => String("०१२३४५६७८९".indexOf(d)));
    const hiTokens = hiNorm.match(/[\u0900-\u097FA-Za-z0-9-]+|[,;:.!?—…"«»“”]+/g) || [];
    const hiOut = [];
    for (const tok of hiTokens) {
      if (/^[,;:.!?—…"«»“”]+$/.test(tok)) {
        if (hiOut.length > 0) hiOut[hiOut.length - 1] += tok;
      } else {
        const ph = phonemizeHindiWord(tok);
        if (ph) hiOut.push(ph);
      }
    }
    return hiOut.join(" ").replace(/\s+/g, " ").trim().normalize("NFD");
  }

  let normalized = text;
  if (lang.startsWith("es")) {
    normalized = normalized
      .replace(/(\d+)\s*%/g, "$1 por ciento")
      .replace(/%/g, " por ciento ")
      .replace(/€\s*(\d+)/g, "$1 euros")
      .replace(/(\d+)\s*€/g, "$1 euros");
  } else if (lang.startsWith("fr")) {
    normalized = normalized
      .replace(/(\d+)\s*%/g, "$1 pour cent")
      .replace(/%/g, " pour cent ")
      .replace(/€\s*(\d+)/g, "$1 euros")
      .replace(/(\d+)\s*€/g, "$1 euros");
  } else if (lang.startsWith("it")) {
    normalized = normalized
      .replace(/(\d+)\s*%/g, "$1 per cento")
      .replace(/%/g, " per cento ")
      .replace(/€\s*(\d+)/g, "$1 euro")
      .replace(/(\d+)\s*€/g, "$1 euro");
  } else {
    normalized = normalized
      .replace(/(\d+)\s*%/g, "$1 por cento")
      .replace(/%/g, " por cento ")
      .replace(/€\s*(\d+)/g, "$1 euros")
      .replace(/(\d+)\s*€/g, "$1 euros");
  }

  // Tokenize into words and punctuation clusters
  const tokens = normalized.match(/[A-Za-zÀ-ÖØ-öø-ÿŒœ0-9'-]+|[,;:.!?¡¿—…"«»“”]+/g) || [];
  const out = [];

  for (let t = 0; t < tokens.length; t++) {
    const tok = tokens[t];
    if (/^[,;:.!?¡¿—…"«»“”]+$/.test(tok)) {
      if (out.length > 0) {
        out[out.length - 1] += tok;
      }
    } else {
      let ph = "";
      if (lang === "pt-BR") {
        ph = phonemizePortugueseBrazilWord(tok);
      } else if (lang === "es-ES") {
        ph = phonemizeSpanishWord(tok, true);
      } else if (lang === "es-419") {
        ph = phonemizeSpanishWord(tok, false);
      } else if (lang === "fr-FR") {
        ph = phonemizeFrenchWord(tok);
      } else if (lang === "it-IT") {
        ph = phonemizeItalianWord(tok);
      } else {
        ph = phonemizePortugueseWord(tok);
      }

      if (ph) {
        // Apply European Portuguese cross-word sibilant sandhi (liaison) when lang === "pt-PT"
        if (lang === "pt-PT" && out.length > 0 && out[out.length - 1].endsWith("ʃ")) {
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
let initPromise = null;
let initPromiseKey = null;

async function getOrInitTTS(requestedMode = "wasm") {
  const wantsWasmQ4 = requestedMode === "wasm-q4" || requestedMode === "q4";
  const wantsWasm = wantsWasmQ4 || requestedMode === "wasm" || requestedMode === "wasm-mobile" || requestedMode === "q8";
  const isMobileWebGPU = requestedMode === "webgpu-mobile";

  // Check if WebGPU is available and backed by real hardware (not a Windows WARP/SwiftShader software fallback)
  let canUseWebGPU = false;
  if (!wantsWasm && typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: isMobileWebGPU ? "low-power" : "high-performance"
      });
      if (adapter && !adapter.isFallbackAdapter) {
        const info = adapter.info || (typeof adapter.requestAdapterInfo === "function" ? await adapter.requestAdapterInfo().catch(() => ({})) : {});
        const desc = `${info.vendor || ""} ${info.architecture || ""} ${info.description || ""} ${info.device || ""}`.toLowerCase();
        const isSoftwareRenderer = info.isFallbackAdapter || /basic render|swiftshader|llvmpipe|software|warp|microsoft basic/.test(desc);
        if (!isSoftwareRenderer) {
          canUseWebGPU = true;
        } else {
          console.warn("Rejected software/fallback WebGPU adapter in favor of fast WASM SIMD:", desc);
        }
      }
    } catch (e) {
      console.warn("Worker WebGPU requestAdapter failed:", e);
      canUseWebGPU = false;
    }
  }

  const targetDevice = canUseWebGPU ? "webgpu" : "wasm";
  const targetDtype = canUseWebGPU ? "fp32" : (wantsWasmQ4 ? "q4" : "q8");
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

  // If an initialization for the exact same engine/dtype is already in-flight (e.g. from background preload), reuse it!
  if (initPromise && initPromiseKey === cacheKey) {
    return initPromise;
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

  initPromiseKey = cacheKey;
  initPromise = (async () => {
    try {
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

      // 2. WebAssembly CPU SIMD with q8 (~85 MB) or q4 (~45 MB) — 4x-8x faster download & zero GPU shader stalls
      const wasmDtype = wantsWasmQ4 ? "q4" : "q8";
      const wasmLabel = wantsWasmQ4 ? "Turbo Mode ⚡⚡ (~45 MB)" : (requestedMode === "wasm-mobile" ? "Mobile Fast Mode 📱" : "Fast Mode ⚡ (~85 MB)");
      self.postMessage({
        status: "progress",
        data: { percent: 0, file: "Connecting...", message: `Initializing Kokoro on ${wasmLabel}...` }
      });

      ttsInstance = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
        dtype: wasmDtype,
        device: "wasm",
        progress_callback: progressCb
      });
      currentDevice = "wasm";
      currentDtype = `wasm-${wasmDtype}`;
      self.postMessage({ status: "device_ready", device: "wasm", dtype: wasmDtype, mobile: requestedMode === "wasm-mobile" });
      return ttsInstance;
    } finally {
      initPromise = null;
      initPromiseKey = null;
    }
  })();

  return initPromise;
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
    const sentences = para.match(/[^.!?।。！？]+[.!?।。！？]+(\s|$)|[^.!?।。！？]+$/g) || [para];
    let current = "";

    for (let sentence of sentences) {
      sentence = sentence.trim();
      if (!sentence) continue;

      if ((current + " " + sentence).trim().length <= maxChars) {
        current = current ? current + " " + sentence : sentence;
      } else {
        if (current) chunks.push(current);
        if (sentence.length > maxChars) {
          const clauses = sentence.match(/[^,;:—、，；：]+[,;:—、，；：]+(\s|$)|[^,;:—、，；：]+$/g) || [sentence];
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
      await getOrInitTTS(dtype || "wasm");
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
      let tts = await getOrInitTTS(dtype || "wasm");
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

      // Only add initial VRAM breather when using WebGPU
      if (currentDevice === "webgpu") {
        await new Promise((resolve) => setTimeout(resolve, isMobileGpu ? 60 : 25));
      }

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

        let res;
        try {
          res = isPortugueseVoice
            ? await generatePortugueseChunk(tts, chunk, selectedVoice, speedValue)
            : await tts.generate(chunk, {
                voice: selectedVoice,
                speed: speedValue
              });
        } catch (chunkErr) {
          // If WebGPU fails during shader execution, automatically fall back to fast WASM q8
          if (currentDevice === "webgpu") {
            console.warn("WebGPU inference failed, switching to WASM Fast Mode:", chunkErr);
            ttsInstance = null;
            currentDtype = null;
            tts = await getOrInitTTS("wasm");
            res = isPortugueseVoice
              ? await generatePortugueseChunk(tts, chunk, selectedVoice, speedValue)
              : await tts.generate(chunk, {
                  voice: selectedVoice,
                  speed: speedValue
                });
          } else {
            throw chunkErr;
          }
        }

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
        if (i < chunks.length - 1) {
          const yieldMs = isMobileGpu ? 140 : (currentDevice === "webgpu" ? 70 : 5);
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
