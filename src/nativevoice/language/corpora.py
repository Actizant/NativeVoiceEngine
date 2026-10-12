"""Small hand-authored multilingual corpus backing the lexicon identifier.

Everything here is written by hand from common function words; it is a domain
prior, not scraped data, and it is deliberately tiny. The point is *not* to be a
state-of-the-art language identifier - it is to give the deterministic, offline
``LexiconLanguageIdentifier`` enough signal to separate, say, a Devanagari Hindi
sentence from a Marathi one, and to recognise the romanised-Hindi pattern that a
pure acoustic classifier silently mislabels as English.

Because the corpus is small, its scores are only ever *relative*: the identifier
must set ``is_calibrated = False`` and never dress a normalised score up as a
probability. Bengali is included on purpose - the reference registry we studied
has no ``bn`` entry at all, which is the exact failure this file exists to avoid.
"""

from __future__ import annotations

# High-frequency function words per language, in that language's own script.
STOPWORDS: dict[str, tuple[str, ...]] = {
    "en": (
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
        "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "from",
        "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all",
        "would", "there", "their", "what", "so", "up", "out", "if", "about", "who",
        "get", "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "your", "good", "some",
        "could", "them", "see", "what", "then", "now", "how", "because", "any",
    ),
    "hi": (
        "और", "है", "हैं", "का", "की", "के", "में", "को", "ने", "से", "पर", "यह",
        "वह", "कि", "जो", "से", "तक", "भी", "या", "नहीं", "था", "थी", "थे", "हो",
        "कर", "लिये", "दिया", "हुआ", "अपने", "अपनी", "एक", "दो", "बहुत", "अच्छा",
        "क्या", "कौन", "कैसे", "कब", "कहाँ", "क्यों", "मुझे", "तुम", "आप", "हम",
        "मेरा", "तेरा", "उनका", "इस", "उस", "कुछ", "सभी", "साथ", "लिए", "वाला",
    ),
    "mr": (
        "आणि", "आहे", "आहोत", "चा", "ची", "चे", "मध्ये", "ला", "ने", "ून", "वर",
        "हा", "ही", "हे", "या", "तो", "ती", "ते", "किंवा", "नाही", "होता", "होती",
        "कर", "द्या", "झाला", "आपले", "तुमचे", "एक", "दोन", "खूप", "चांगले",
        "काय", "कोण", "कसे", "केव्हा", "कुठे", "का", "मला", "तुम्ही", "आम्ही",
        "म्हणून", "सोबत", "साठी", "असले", "असणार", "ज्या", "याचा", "अधिक",
    ),
    "bn": (
        "এবং", "হয়", "হল", "এক", "দুই", "অনেক", "আমি", "তুমি", "সে", "আমরা",
        "তোমরা", "তারা", "এটা", "এটি", "সেটা", "কী", "কে", "কখন", "কোথায়",
        "কেন", "কিভাবে", "না", "হয়নি", "ছিল", "হবে", "কর", "করে", "করা",
        "তার", "আমার", "তোমার", "নিজের", "এ", "ও", "ওর", "খুব", "ভালো",
        "একের", "জন্য", "সাথে", "থেকে", "পর্যন্ত", "মধ্যে", "উপর", "ওপর", "প্রতি",
    ),
    "ta": (
        "மற்றும்", "ஆம்", "இல்லை", "ஒரு", "இரண்டு", "நிறைய", "நான்", "நீ",
        "அவன்", "அவள்", "அது", "இது", "என்ன", "யார்", "எப்போது", "எங்கே",
        "ஏன்", "எப்படி", "இருந்தது", "ஆகும்", "செய்", "செய்து", "செய்ய",
        "அவரது", "எனது", "உனது", "மிக", "நல்ல", "ஒன்றாக", "க்காக", "உடன்",
        "இருந்து", "வரை", "மீது", "பல", "சில", "வேறு", "குறிப்பாக", "உண்மையில்",
    ),
    "te": (
        "మరియు", "కాదు", "ఓక", "రెండు", "చాలా", "నేను", "నీవు", "అతడు",
        "ఆమె", "అది", "ఇది", "ఏమిటి", "ఎవరు", "ఎప్పుడు", "ఎక్కడ", "ఎందుకు",
        "ఎలా", "ఉంది", "ఉన్నది", "కొని", "చేసి", "చేయ", "అతని", "నా",
        "నీ", "చాల", "మంచి", "కొరకు", "తొ", "నుండి", "వరకు", "పై", "కొన్ని",
        "వేరే", "నిజంగా", "ముఖ్యంగా", "ఈ", "ఆ", "ఆయన", "మనం", "మీరు",
    ),
    "ur": (
        "اور", "ہے", "ہیں", "کا", "کی", "کے", "میں", "کو", "نے", "سے", "پر",
        "یہ", "وہ", "کہ", "جو", "تک", "بھی", "یا", "نہیں", "تھا", "تھی", "ہوا",
        "ہو", "کر", "لیے", "دیا", "اپنے", "اپنی", "ایک", "دو", "بہت", "اچھا",
        "کیا", "کون", "کیسے", "کب", "کہاں", "کیوں", "مجھے", "تم", "آپ", "ہم",
        "میرا", "اس", "تو", "سب", "کچھ", "ساتھ", "والا",
    ),
    "es": (
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
        "por", "un", "para", "con", "no", "una", "su", "al", "como", "más",
        "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre",
        "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay",
        "donde", "quien", "después", "todo", "nos", "durante", "todos", "uno",
        "les", "ni", "contra", "otros", "eso", "ante", "ellos", "mí", "cómo",
    ),
    "fr": (
        "de", "des", "le", "la", "et", "les", "un", "une", "du", "être", "avoir",
        "à", "au", "aux", "en", "dans", "que", "qui", "ce", "cette", "il",
        "elle", "nous", "vous", "ils", "ne", "pas", "plus", "pour", "par",
        "avec", "sur", "sous", "entre", "comme", "mais", "ou", "donc", "car",
        "si", "quand", "comment", "pourquoi", "tout", "tous", "toute", "même",
        "aussi", "bien", "très", "peu", "beaucoup", "ici", "là", "avant", "après",
    ),
    "de": (
        "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "auf",
        "für", "ist", "im", "dem", "nicht", "aus", "eine", "noch", "nach", "wie",
        "auch", "nur", "ein", "wir", "du", "sie", "er", "es", "an", "werden",
        "aber", "mir", "mich", "dann", "schon", "weil", "kein", "ob", "und",
        "über", "unter", "vor", "hinter", "zwischen", "immer", "etwas", "viel",
        "mehr", "sehr", "hier", "dort", "wann", "warum", "wie", "was", "wer",
    ),
    "ja": (
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
        "ある", "いる", "も", "する", "から", "な", "こと", "として", "ない",
        "これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ",
        "私", "あなた", "彼", "彼女", "僕", "何", "いつ", "どこ", "なぜ", "どう",
        "です", "ます", "でした", "ました", "います", "いました", "います",
    ),
    "zh": (
        "的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "我们", "你们",
        "他们", "这", "那", "什么", "谁", "哪", "怎么", "为什么", "不", "没",
        "有", "没有", "一", "二", "三", "很", "也", "都", "就", "还", "又", "才",
        "已经", "可以", "会", "要", "想", "说", "做", "用", "过", "给", "让", "把",
        "被", "对", "与", "但", "因为", "所以", "如果", "这个", "那个", "时候",
    ),
    "ar": (
        "ال", "من", "إلى", "على", "في", "عن", "هذا", "هذه", "ذلك", "الذي",
        "التي", "ما", "ماذا", "كيف", "متى", "أين", "لماذا", "لا", "نعم", "هو",
        "هي", "نحن", "أنتم", "هم", "كان", "يكون", "قد", "سي", "كل", "بعض",
        "آخر", "جدا", "بين", "ثم", "أو", "و", "أن", "إن", "لكن", "حتى", "منذ",
        "عند", "فوق", "تحت", "مع", "بدون", "مثل", "قبل", "بعد", "كثير", "قليل",
    ),
}

# Common characters that appear in Indic scripts beyond the function words.
CHAR_NGRAM_SEED: dict[str, tuple[str, ...]] = {
    "hi": ("्", "म", "न", "र", "स", "क", "ह", "ी", "े", "ा", "ब", "प"),
    "mr": ("्", "म", "न", "र", "आ", "ह", "ी", "े", "ा", "च", "छ", "झ"),
    "bn": ("্", "ম", "ন", "র", "স", "হ", "ী", "ে", "া", "ব", "প", "য়"),
    "ta": ("்", "ம", "ந", "ர", "ச", "க", "ந", "ீ", "ே", "ா", "ப", "ல"),
    "te": ("్", "మ", "న", "ర", "స", "క", "హ", "ీ", "ే", "ా", "ప", "ల"),
    "ur": ("ے", "ہ", "ی", "ک", "ر", "م", "ن", "س", "ل", "ت", "ب", "د"),
    "ar": ("ة", "ى", "ي", "ر", "م", "ن", "س", "ل", "ت", "ب", "د", "إ"),
    "ja": ("の", "に", "は", "を", "が", "で", "た", "て", "し", "れ", "い", "う"),
    "zh": ("的", "了", "是", "在", "我", "有", "和", "人", "这", "中", "大", "为"),
}

# Words that betray Roman Urdu when the dominant script is Latin. Chosen to be
# almost never English words so their presence is a strong transliteration cue.
ROMAN_URDU_MARKERS: tuple[str, ...] = (
    "kya", "kyu", "kyon", "kyun", "kaise", "kaisa", "kaisi", "hai", "hain",
    "nahi", "nahin", "karo", "karna", "kar", "kariye", "kiya", "aap", "tum",
    "tera", "apna", "apni", "mujhe", "mujhy", "mujhko", "bahut", "bohat",
    "acha", "accha", "theek", "thik", "thoda", "thora", "paani", "aur", "bhi",
    "wala", "wali", "kal", "aaj", "rozi", "zaroor", "bilkul", "shayad",
    "khana", "khana", "chaiye", "chahiye", "chahye", "sakte", "skte", "raha",
    "rahi", "rahe", "gaya", "gayi", "hoga", "hogi", "hota", "hoti", "batao",
    "bataye", "sunao", "samjho", "samjha", "dekho", "dekho", "jao", "aao",
    "aana", "jaana", "zyada", "jada", "thak", "pehle", "baad", "dvara", "dwara",
)

# Explicit Hindi markers, largely overlapping Roman Urdu - the collision is real.
ROMAN_HINDI_MARKERS: tuple[str, ...] = (
    "kaise", "kaisa", "kyun", "kyon", "nahi", "nahin", "hai", "hain", "tha",
    "thi", "the", "hoon", "hoga", "kar", "karo", "karna", "mujhe", "aap",
    "tum", "tera", "apna", "bahut", "acha", "accha", "thik", "theek", "thoda",
    "paani", "wala", "wali", "bhi", "aur", "kyunki", "jab", "tab", "yahan",
    "wahan", "kahan", "kya", "kaun", "kab", "kaise", "kitna", "kitni", "zyada",
    "jada", "pehle", "baad", "dvara", "dwara", "sakte", "raha", "rahi", "rahe",
    "gaya", "gayi", "hota", "hoti", "batao", "samjho", "dekho", "zyada",
)

# A few complete sentences per language, used to verify the corpus at import and
# referenced by the test suite. They are illustrative, not training data.
SAMPLE_SENTENCES: dict[str, str] = {
    "en": "the quick brown fox jumps over the lazy dog and then it sleeps",
    "hi": "मैं ठीक हूँ और आप कैसे हैं कृपया मुझे थोड़ा पानी दीजिए",
    "bn": "আমি ভালো আছি আপনি কেমন আছেন অনুগ্রহ করে আমাকে একটু জল দিন",
    "ta": "நான் நலமாக இருக்கிறேன் நீங்கள் எப்படி இருக்கிறீர்கள் தயவுசெய்து தண்ணீர் தரவும்",
    "te": "నేను బాగున్నాను మీరు ఎలా ఉన్నారు దయచేసి కొంచెం నీరు ఇవ్వండి",
    "mr": "मी ठीक आहे आणि तुम्ही कसे आहात कृपया मला थोडे पाणी द्या",
    "ur": "میں ٹھیک ہوں اور آپ کیسے ہیں براہ کرم مجھے تھوڑا پانی دیں",
    "es": "el zorro marrón rápido salta sobre el perro perezoso y luego duerme",
    "fr": "le rapide renard brun saute par-dessus le chien paresseux puis il dort",
    "de": "der schnelle braune füchse springt über den faulen hund und schläft dann",
    "ja": "すばやい褐色のキツネが怠惰な犬の上を飛び越えてその後眠ります",
    "zh": "敏捷的棕色狐狸跳过懒狗然后睡着了",
    "ar": "الكلب الكسول والنمر السريع على سطح الحديقة المائلة",
}

# Indic scripts that a lexicon entry can already disambiguate by itself.
LEXICON_SCRIPTS: dict[str, str] = {
    "hi": "Devanagari",
    "mr": "Devanagari",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Arabic",
    "ar": "Arabic",
    "he": "Hebrew",
    "ru": "Cyrillic",
    "ja": "Hiragana",
    "zh": "Han",
    "ko": "Hangul",
    "el": "Greek",
    "th": "Thai",
    "km": "Khmer",
    "my": "Myanmar",
    "lo": "Lao",
    "am": "Ethiopic",
    "ka": "Georgian",
    "hy": "Armenian",
}
