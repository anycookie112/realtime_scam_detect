#!/usr/bin/env python3
"""Generate 100+ test scenarios per language with controlled variation.

Outputs JSON files compatible with the existing TTS pipeline:
    test_data/en.json     — 100 English scenarios
    test_data/ms.json     — 100 Bahasa Malaysia
    test_data/mixed.json  — 100 Manglish (code-switched)
    test_data/zh.json     — 100 Mandarin

Each scenario is a dict with:
    name:        unique identifier (used as audio filename)
    text:        the transcript
    risk_level:  HIGH_RISK | MEDIUM_RISK | LOW_RISK | SAFE
    category:    bank_impersonation | gov_impersonation | telco | delivery |
                 investment | romance | job_loan | legit_bank | legit_service |
                 suspicious_sales

Usage:
    uv run tools/tts/generate_test_data.py
    uv run tools/tts/generate_test_data.py --lang en --count 200
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

random.seed(42)  # reproducible

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "test_data"

# ── Variation slots ─────────────────────────────────────────────────────────

BANKS = [
    {"name_en": "Maybank",       "name_ms": "Maybank",       "name_zh": "马来亚银行",
     "alias": "MBB",             "hotline": "1-300-88-6688"},
    {"name_en": "Public Bank",   "name_ms": "Public Bank",   "name_zh": "大众银行",
     "alias": "PBB",             "hotline": "1-800-22-5555"},
    {"name_en": "CIMB",          "name_ms": "CIMB",          "name_zh": "联昌银行",
     "alias": "CIMB",            "hotline": "03-6204-7788"},
    {"name_en": "RHB Bank",      "name_ms": "RHB Bank",      "name_zh": "兴业银行",
     "alias": "RHB",             "hotline": "03-9206-8118"},
    {"name_en": "Hong Leong",    "name_ms": "Hong Leong",    "name_zh": "丰隆银行",
     "alias": "HLB",             "hotline": "03-7626-8899"},
    {"name_en": "AmBank",        "name_ms": "AmBank",        "name_zh": "大马银行",
     "alias": "AmBank",          "hotline": "03-2178-8888"},
    {"name_en": "Bank Islam",    "name_ms": "Bank Islam",    "name_zh": "回教银行",
     "alias": "BIMB",            "hotline": "03-2609-0900"},
    {"name_en": "BSN",           "name_ms": "BSN",           "name_zh": "国家储蓄银行",
     "alias": "BSN",             "hotline": "1-300-88-1900"},
]

AMOUNTS = ["RM 450", "RM 890", "RM 1,250", "RM 2,300", "RM 3,500", "RM 5,800",
           "RM 8,900", "RM 12,400", "RM 15,000", "RM 25,000", "RM 50,000"]

CARD_DIGITS = ["4523", "7736", "8871", "9234", "1156", "3478", "6612", "5589"]

MERCHANTS_EN = ["jewellery store", "electronics shop", "petrol station",
                "online shop", "supermarket", "restaurant", "luxury boutique"]
MERCHANTS_MS = ["kedai barang kemas", "kedai elektronik", "stesen minyak",
                "kedai dalam talian", "pasar raya", "restoran", "butik mewah"]
MERCHANTS_ZH = ["珠宝店", "电器店", "加油站", "网店", "超市", "餐厅", "名牌店"]

CITIES_EN = ["Kuala Lumpur", "Johor Bahru", "Penang", "Ipoh", "Shah Alam",
             "Petaling Jaya", "Melaka", "Kuching"]
CITIES_MS = CITIES_EN  # same names
CITIES_ZH = ["吉隆坡", "新山", "槟城", "怡保", "莎阿南", "八打灵再也", "马六甲", "古晋"]

ACCOUNT_NUMBERS = ["5628-1190-3347", "7812-3344-5566", "9001-2233-4455",
                   "1234-5678-9012", "8899-7766-5544"]

GOV_AGENCIES_EN = [
    {"name": "PDRM Bukit Aman", "type": "police"},
    {"name": "LHDN", "type": "tax"},
    {"name": "Bank Negara", "type": "central_bank"},
    {"name": "Pos Laju customs", "type": "delivery"},
    {"name": "MCMC", "type": "telco_regulator"},
    {"name": "JPN", "type": "registration"},
]

TELCOS = ["Maxis", "Celcom", "Digi", "U Mobile", "Unifi", "TM"]

DELIVERY_COMPANIES = ["Pos Laju", "J&T Express", "Lazada", "Shopee",
                      "Ninja Van", "DHL", "FedEx"]

NAMES_EN = ["Sarah", "Ahmad", "Lim", "Mei Ling", "Hassan", "Vivian",
            "Amir", "Catherine", "Daniel", "Priya"]

# ── English templates ──────────────────────────────────────────────────────

HIGH_RISK_TEMPLATES_EN = [
    # Bank impersonation — credential phishing
    "Hello, this is {bank} security department. We detected unauthorised access on your {bank2u} account. Please share your full IC number and online banking password immediately so we can secure your account.",
    "Good afternoon, calling from {bank} fraud team. Your account is at risk. We need you to share the OTP that was just sent to your phone to verify your identity.",
    "I'm from {bank} compliance department. To prevent your account from being frozen, please share your Secure2u approval code right now.",
    "This is {bank}. We're transferring you to our fraud officer. To proceed, please provide your full {bank} card number and the CVV code on the back.",

    # Bank impersonation — transfer to safe account
    "Hello, this is {bank} fraud monitoring. Your account has been compromised. For your safety, please transfer all funds to our secure holding account: {account}.",
    "Calling from {bank}. Suspicious activity detected. Please move your savings to this temporary protected account: {account}. Don't tell anyone about this call.",
    "{bank} security alert. Your money is at risk. Transfer it immediately to {account} for safekeeping while we investigate.",

    # Police/PDRM impersonation
    "This is Sergeant {name} from {agency}. Your IC has been used for money laundering. There's a warrant for your arrest. Pay {amount} bail to {account} immediately or officers will come to your house.",
    "Inspector {name} from PDRM. Your bank account is linked to drug trafficking. To clear your name, transfer {amount} to {account} now.",
    "This is {agency}. You are being investigated for criminal activity. To avoid arrest, you must pay a bail of {amount} immediately.",

    # LHDN tax scam
    "Important call from LHDN. You have outstanding tax of {amount}. Pay within 24 hours or warrant will be issued. Press 1 to speak with our enforcement officer now.",
    "LHDN audit department. Your business has unpaid taxes of {amount}. To avoid asset seizure, transfer the payment immediately to {account}.",

    # Investment scam
    "Congratulations! You've been selected for an exclusive forex programme guaranteeing 30 percent monthly returns. Just deposit {amount} to {account} to secure your spot. This offer expires today.",
    "I'm a senior crypto analyst. We have insider info on a token that will 10x this week. Minimum investment is {amount}. Wallet address coming on WhatsApp.",
    "Hi, this is {name} from a private investment fund. Members made {amount} last month. Guaranteed returns, no risk. Send {amount} to {account} to start.",

    # Loan scam
    "Good news! Your personal loan of {amount} has been pre-approved. To release funds today, just pay a processing fee of RM 500 to {account}. Limited time offer.",
    "Your loan application is approved. Transfer the insurance fee of RM 800 to {account} within the hour or the approval will be cancelled.",

    # Telco suspension scam
    "Hello, this is {telco}. Your line has been flagged for illegal activities and will be terminated in 2 hours. Provide your IC and pay {amount} security deposit to {account} to prevent suspension.",
    "{telco} customer service. Your number is being used for criminal activity. Verify your identity by sharing your IC, MyKad number, and bank details now.",

    # Parcel/delivery scam
    "Hello, this is {delivery}. A parcel under your name was intercepted at customs containing illegal items. I'm transferring you to the investigating officer. You need to pay a fine of {amount} to {account} to clear your name.",
    "{delivery} customs department. Your parcel contains contraband. To avoid criminal charges, transfer the inspection fee of {amount} to {account} immediately.",

    # Romance scam
    "Baby, I'm stuck at the airport in London and my wallet was stolen. I need {amount} urgently to buy a ticket home. Please transfer to {account}. I promise I'll pay you back. Don't tell anyone, I'm so embarrassed.",
    "Honey, my mother is in the hospital and I can't access my account. Can you transfer {amount} to {account} to help with her medical bills? I love you and will pay you back next week.",

    # WhatsApp/MAE phishing
    "This is {bank} digital banking. Your {app} app needs to be updated. Click the link I'm sending to your SMS to download the secure update. You'll need to re-enter your username and password to verify.",
    "{bank} security update. Install the new app from this link I just sent you. Then share the verification code that appears on your screen.",

    # Remote access app
    "I'm from {bank} technical support. To fix your account issue, please download TeamViewer and give me the access code. We need to remotely check your phone.",
    "{bank} IT department. Install AnyDesk now so we can resolve the security alert on your device. Share the 9-digit code that appears.",

    # Bogus bank intentional
    "Good morning, Bogus Bank security team. Suspicious activity on your account. I need your full username and password to verify your identity. This is standard procedure.",
    "Bogus Bank calling. Please share your 6-digit OTP and install TeamViewer so we can secure your account. Don't tell anyone about this call.",
]

MEDIUM_RISK_TEMPLATES_EN = [
    # Aggressive sales
    "This is {company} Credit. Your credit card payment of {amount} is overdue by 45 days. If payment is not received by Friday, we'll take legal action and report to CTOS. Commit to a date now.",
    "Hi, I'm from {company} insurance. Your current coverage is very inadequate. If anything happens, your family will be in trouble. I have a plan for {amount} per month. I need your answer today, the rate expires tonight.",
    "{telco} retention department. I see you want to port your number out. Stay with us — I can offer 50 percent off for 6 months, but I need your MyKad number to process this right now.",

    # Unsolicited callback request
    "Hello, this is {bank}. We noticed unusual login attempts from a new device. For security, please call us back at {hotline} within 30 minutes. Have your account number and IC ready.",
    "{bank} fraud team. Suspicious activity detected. Please call us back urgently at {hotline}. Don't delay or your account may be locked.",

    # Aggressive surveys
    "I'm conducting a financial wellness survey for Bank Negara. I need your monthly income, total savings, number of accounts, and outstanding loans. Can I also get your full name and IC number?",
    "Government health survey. We need your full name, IC, address, and household income for national statistics. The survey takes 10 minutes.",

    # Unverifiable charity
    "Good evening, calling from a Malaysian Red Crescent fundraiser for flood relief in Kelantan. Would you like to donate? I can give you our account number now. For donations of {amount} or more, we'll send a tax-exempt receipt.",
    "Hi, this is the Children's Aid Foundation. We're raising funds for orphans. Can you donate {amount} to our account: {account}? Every ringgit helps.",

    # Suspicious cold sales
    "I'm calling about a special timeshare opportunity. Limited slots, exclusive offer, but I need your decision today. Can I get your IC and bank details to reserve?",
    "Hi, you've been selected for a luxury holiday package worth {amount}. Just pay the booking fee of RM 500 to confirm. The offer expires at 6pm today.",

    # Persistent debt collector — borderline
    "This is the third notice from {company} Credit Recovery. Your debt of {amount} must be settled by Monday or we proceed to court. What's your payment plan?",
    "Final warning. Your unpaid bill of {amount} will be referred to collections agency tomorrow. Make payment now to avoid legal action.",
]

LOW_RISK_TEMPLATES_EN = [
    # Promotional calls
    "Hi, calling from {bank} card centre. We have an exclusive zero percent instalment plan for 12 months on purchases above {amount} at selected merchants. Would you like to hear more?",
    "{telco} customer service. We have a new plan with 100GB data and unlimited calls for RM 80 monthly, which is RM 20 less than your current plan. No commitment. Interested?",
    "{bank} Premier service. We'd like to offer you a complimentary financial planning session at our flagship branch. Are you free next week?",

    # Customer satisfaction surveys
    "Calling from {bank} customer experience team. You recently visited our {city} branch. We'd appreciate a 2-minute feedback survey. We won't ask for personal details. Skip any question you're not comfortable with.",
    "Hi, this is a quick feedback call from {company}. How would you rate your recent experience with us on a scale of 1 to 5?",

    # Service appointment reminders
    "Reminder: your appointment with {bank} relationship manager is at 2pm tomorrow at our {city} branch. Please bring your IC. To reschedule, call {hotline}.",
    "Hello, this is your insurance agent. Just a reminder your annual policy review is next week. No payment needed at this stage.",
]

SAFE_TEMPLATES_EN = [
    # Legitimate fraud alert
    "Hello, calling from {bank} regarding a transaction on your card ending {digits}. There was a charge of {amount} at a {merchant} in {city} today. Was this you? If not, we'll block the card and you can visit any branch with your IC.",
    "{bank} fraud monitoring. Your card ending {digits} was used for {amount} at a {merchant} at 3pm. Did you make this purchase? If not, we'll freeze the card immediately.",

    # Payment reminder
    "Hello, courtesy call from {bank}. Your home loan instalment of {amount} is due on the 20th. If you've already paid, please disregard. For questions call {hotline} or visit any branch.",
    "{bank} reminder: your credit card minimum payment of {amount} is due in 3 days. Payment can be made via {bank2u} or any branch.",

    # Standard verification
    "{bank} calling about your recent application. Could you confirm the last 4 digits of your IC for verification? We don't need any other personal information.",
    "Hello from {bank}. To process your card replacement request, please confirm your name and the last 4 digits of your account. Thank you.",

    # Delivery notification
    "Hello, this is {delivery}. We have a parcel for you, tracking number {tracking}. We tried to deliver today but no one was home. Would you like redelivery tomorrow 9am to 12pm, or pickup from our {city} hub?",
    "{delivery} dispatch. Your package is out for delivery today. The driver will call before arrival. No payment needed.",

    # Insurance renewal
    "Good morning from AIA Malaysia regarding your life insurance policy ending {digits}. Your premium of {amount} is due next month on the 15th. Just confirming your contact details. You can manage your policy through the AIA app.",

    # Branch appointment
    "{bank} Premier. Just a reminder you have an appointment with your relationship manager Ms. {name} at our {city} branch tomorrow at 2pm. Please bring your IC. To reschedule, call {hotline}.",

    # Transaction confirmation
    "{bank} confirming your fund transfer of {amount} from your account ending {digits}. If you didn't authorize this, please call {hotline} immediately.",
]


# ── Template translations to other languages ────────────────────────────────
# We provide the same scenario templates translated. The slot fillers above
# are language-aware (e.g., MERCHANTS_MS for Malay).

HIGH_RISK_TEMPLATES_MS = [
    "Hello, ini jabatan keselamatan {bank}. Kami mengesan akses tanpa kebenaran pada akaun {bank2u} anda. Sila kongsikan nombor IC penuh dan kata laluan perbankan dalam talian dengan segera untuk kami melindungi akaun anda.",
    "Selamat petang, ini pasukan penipuan {bank}. Akaun anda dalam bahaya. Kami perlukan anda kongsikan OTP yang baru dihantar ke telefon anda untuk pengesahan identiti.",
    "Saya dari jabatan pematuhan {bank}. Untuk mengelakkan akaun anda dibekukan, sila kongsikan kod kelulusan Secure2u sekarang.",
    "Ini {bank}. Kami akan memindahkan anda kepada pegawai penipuan. Untuk meneruskan, sila berikan nombor kad {bank} penuh dan kod CVV di belakang kad.",

    "Hello, ini pemantauan penipuan {bank}. Akaun anda telah terjejas. Untuk keselamatan anda, sila pindahkan semua wang ke akaun pegangan selamat kami: {account}.",
    "Menghubungi dari {bank}. Aktiviti mencurigakan dikesan. Sila pindahkan simpanan anda ke akaun perlindungan sementara ini: {account}. Jangan beritahu sesiapa tentang panggilan ini.",
    "Amaran keselamatan {bank}. Wang anda dalam bahaya. Pindahkan segera ke {account} untuk simpanan selamat semasa kami menyiasat.",

    "Ini Sarjan {name} dari {agency}. IC anda telah digunakan untuk pengubahan wang haram. Ada waran tangkap untuk anda. Bayar jaminan {amount} ke {account} segera atau pegawai akan datang ke rumah anda.",
    "Inspektor {name} dari PDRM. Akaun bank anda dikaitkan dengan pengedaran dadah. Untuk membersihkan nama anda, pindahkan {amount} ke {account} sekarang.",
    "Ini {agency}. Anda sedang disiasat untuk aktiviti jenayah. Untuk mengelakkan tangkapan, anda mesti bayar jaminan {amount} segera.",

    "Panggilan penting dari LHDN. Anda mempunyai cukai tertunggak sebanyak {amount}. Bayar dalam 24 jam atau waran akan dikeluarkan. Tekan 1 untuk bercakap dengan pegawai penguatkuasaan kami sekarang.",
    "Jabatan audit LHDN. Perniagaan anda mempunyai cukai tidak dibayar sebanyak {amount}. Untuk mengelakkan rampasan aset, pindahkan bayaran segera ke {account}.",

    "Tahniah! Anda telah terpilih untuk program forex eksklusif yang menjamin pulangan 30 peratus sebulan. Hanya deposit {amount} ke {account} untuk merebut tempat anda. Tawaran ini tamat hari ini.",
    "Saya penganalisis kripto kanan. Kami ada maklumat dalaman tentang token yang akan 10x minggu ini. Pelaburan minimum {amount}. Alamat dompet akan dihantar di WhatsApp.",
    "Hi, ini {name} dari dana pelaburan persendirian. Ahli-ahli buat {amount} bulan lepas. Pulangan terjamin, tiada risiko. Hantar {amount} ke {account} untuk mula.",

    "Berita baik! Pinjaman peribadi anda sebanyak {amount} telah pra-diluluskan. Untuk pengeluaran dana hari ini, hanya bayar yuran pemprosesan RM 500 ke {account}. Tawaran masa terhad.",
    "Permohonan pinjaman anda diluluskan. Pindahkan yuran insurans RM 800 ke {account} dalam masa sejam atau kelulusan akan dibatalkan.",

    "Hello, ini {telco}. Talian anda telah dibenderakan untuk aktiviti haram dan akan ditamatkan dalam 2 jam. Berikan IC anda dan bayar deposit keselamatan {amount} ke {account} untuk mengelakkan penggantungan.",
    "Khidmat pelanggan {telco}. Nombor anda digunakan untuk aktiviti jenayah. Sahkan identiti dengan kongsikan IC, nombor MyKad, dan butiran bank sekarang.",

    "Hello, ini {delivery}. Bungkusan atas nama anda telah dirampas di kastam kerana mengandungi barang haram. Saya akan memindahkan anda kepada pegawai penyiasat. Anda perlu bayar denda {amount} ke {account} untuk membersihkan nama anda.",
    "Jabatan kastam {delivery}. Bungkusan anda mengandungi barang haram. Untuk mengelakkan tuduhan jenayah, pindahkan yuran pemeriksaan {amount} ke {account} segera.",

    "Sayang, saya tersekat di lapangan terbang London dan dompet saya hilang. Saya perlukan {amount} segera untuk beli tiket pulang. Tolong pindahkan ke {account}. Saya janji akan bayar balik. Jangan beritahu sesiapa, saya malu.",
    "Sayang, ibu saya di hospital dan saya tak boleh akses akaun. Boleh awak pindahkan {amount} ke {account} untuk bantu bayar bil perubatan? Saya cinta awak dan akan bayar balik minggu depan.",

    "Ini sokongan perbankan digital {bank}. Aplikasi {app} anda perlu dikemas kini. Klik pautan yang saya hantar ke SMS untuk muat turun kemas kini selamat. Anda perlu masukkan semula nama pengguna dan kata laluan untuk pengesahan.",
    "Kemas kini keselamatan {bank}. Pasang aplikasi baru dari pautan yang saya baru hantar. Kemudian kongsikan kod pengesahan yang muncul di skrin anda.",

    "Saya dari sokongan teknikal {bank}. Untuk membaiki masalah akaun anda, sila muat turun TeamViewer dan beri saya kod akses. Kami perlu memeriksa telefon anda dari jauh.",
    "Jabatan IT {bank}. Pasang AnyDesk sekarang supaya kami boleh menyelesaikan amaran keselamatan pada peranti anda. Kongsikan kod 9 digit yang muncul.",

    "Selamat pagi, pasukan keselamatan Bogus Bank. Aktiviti mencurigakan pada akaun anda. Saya perlukan nama pengguna dan kata laluan penuh anda untuk pengesahan. Ini prosedur standard.",
    "Bogus Bank menghubungi. Sila kongsikan OTP 6 digit anda dan pasang TeamViewer supaya kami boleh melindungi akaun anda. Jangan beritahu sesiapa tentang panggilan ini.",
]

MEDIUM_RISK_TEMPLATES_MS = [
    "Ini {company} Credit. Bayaran kad kredit anda sebanyak {amount} telah lewat 45 hari. Jika tidak diterima menjelang Jumaat, kami akan ambil tindakan undang-undang dan laporkan ke CTOS. Berjanji tarikh sekarang.",
    "Hi, saya dari insurans {company}. Perlindungan semasa anda sangat tidak mencukupi. Jika berlaku apa-apa, keluarga anda akan dalam masalah. Saya ada pelan untuk {amount} sebulan. Saya perlukan jawapan anda hari ini, kadar ini tamat malam ini.",
    "Jabatan retensi {telco}. Saya nampak anda nak port nombor anda keluar. Kekal dengan kami — saya boleh tawarkan diskaun 50 peratus untuk 6 bulan, tapi saya perlukan nombor MyKad anda untuk memprosesnya sekarang.",

    "Hello, ini {bank}. Kami perasan percubaan log masuk luar biasa dari peranti baru. Untuk keselamatan, sila hubungi kami semula di {hotline} dalam masa 30 minit. Sediakan nombor akaun dan IC anda.",
    "Pasukan penipuan {bank}. Aktiviti mencurigakan dikesan. Sila hubungi kami semula segera di {hotline}. Jangan tangguh atau akaun anda mungkin dikunci.",

    "Saya menjalankan tinjauan kewangan untuk Bank Negara. Saya perlukan pendapatan bulanan, jumlah simpanan, bilangan akaun, dan pinjaman tertunggak anda. Boleh saya juga ambil nama penuh dan nombor IC anda?",
    "Tinjauan kesihatan kerajaan. Kami perlukan nama penuh, IC, alamat, dan pendapatan isi rumah anda untuk statistik nasional. Tinjauan ambil masa 10 minit.",

    "Selamat petang, dari pengumpul dana Bulan Sabit Merah Malaysia untuk bantuan banjir di Kelantan. Adakah anda ingin menderma? Saya boleh berikan nombor akaun kami sekarang. Untuk derma {amount} atau lebih, kami akan hantar resit pengecualian cukai.",
    "Hi, ini Yayasan Bantuan Kanak-Kanak. Kami mengumpul dana untuk anak yatim. Boleh anda derma {amount} ke akaun kami: {account}? Setiap ringgit membantu.",

    "Saya menghubungi tentang peluang timeshare istimewa. Slot terhad, tawaran eksklusif, tapi saya perlukan keputusan anda hari ini. Boleh saya dapatkan IC dan butiran bank untuk merezab?",
    "Hi, anda telah terpilih untuk pakej percutian mewah bernilai {amount}. Hanya bayar yuran tempahan RM 500 untuk mengesahkan. Tawaran tamat 6 petang hari ini.",

    "Ini notis ketiga dari {company} Pemulihan Kredit. Hutang anda sebanyak {amount} mesti diselesaikan menjelang Isnin atau kami teruskan ke mahkamah. Apa pelan bayaran anda?",
    "Amaran terakhir. Bil belum dibayar anda sebanyak {amount} akan dirujuk ke agensi pengutipan esok. Buat bayaran sekarang untuk mengelakkan tindakan undang-undang.",
]

LOW_RISK_TEMPLATES_MS = [
    "Hi, menghubungi dari pusat kad {bank}. Kami ada pelan ansuran sifar peratus eksklusif untuk 12 bulan untuk pembelian melebihi {amount} di pedagang terpilih. Anda nak dengar lagi?",
    "Khidmat pelanggan {telco}. Kami ada pelan baru dengan data 100GB dan panggilan tanpa had untuk RM 80 sebulan, iaitu RM 20 lebih murah dari pelan semasa anda. Tiada komitmen. Berminat?",
    "Khidmat Premier {bank}. Kami ingin menawarkan sesi perancangan kewangan percuma di cawangan utama kami. Anda lapang minggu depan?",

    "Menghubungi dari pasukan pengalaman pelanggan {bank}. Anda baru-baru ini melawat cawangan {city} kami. Kami menghargai tinjauan maklum balas 2 minit. Kami tidak akan tanya butiran peribadi. Lompat soalan yang anda tidak selesa.",
    "Hi, ini panggilan maklum balas pantas dari {company}. Bagaimana anda menilai pengalaman terkini anda dengan kami pada skala 1 hingga 5?",

    "Peringatan: temujanji anda dengan pengurus hubungan {bank} adalah pukul 2 petang esok di cawangan {city} kami. Sila bawa IC anda. Untuk menjadual semula, hubungi {hotline}.",
    "Hello, ini ejen insurans anda. Hanya peringatan kajian semula polisi tahunan anda minggu depan. Tiada bayaran diperlukan pada peringkat ini.",
]

SAFE_TEMPLATES_MS = [
    "Hello, menghubungi dari {bank} mengenai transaksi pada kad anda yang berakhir {digits}. Terdapat caj {amount} di {merchant} di {city} hari ini. Adakah ini anda? Jika tidak, kami akan sekat kad dan anda boleh ke mana-mana cawangan dengan IC anda.",
    "Pemantauan penipuan {bank}. Kad anda yang berakhir {digits} digunakan untuk {amount} di {merchant} pada pukul 3 petang. Adakah anda buat pembelian ini? Jika tidak, kami akan bekukan kad segera.",

    "Hello, panggilan ihsan dari {bank}. Ansuran pinjaman rumah anda sebanyak {amount} akan dibayar pada 20hb. Jika anda telah membayar, sila abaikan. Untuk pertanyaan hubungi {hotline} atau lawati mana-mana cawangan.",
    "Peringatan {bank}: bayaran minimum kad kredit anda sebanyak {amount} akan dibayar dalam 3 hari. Bayaran boleh dibuat melalui {bank2u} atau mana-mana cawangan.",

    "{bank} menghubungi mengenai permohonan baru anda. Boleh anda sahkan 4 digit terakhir IC untuk pengesahan? Kami tidak perlukan maklumat peribadi lain.",
    "Hello dari {bank}. Untuk memproses permintaan penggantian kad anda, sila sahkan nama anda dan 4 digit terakhir akaun. Terima kasih.",

    "Hello, ini {delivery}. Kami ada bungkusan untuk anda, nombor jejak {tracking}. Kami cuba hantar hari ini tapi tiada sesiapa di rumah. Anda mahu hantar semula esok 9 pagi hingga 12 tengah hari, atau ambil dari hab {city} kami?",
    "Penghantaran {delivery}. Bungkusan anda dalam perjalanan hari ini. Pemandu akan hubungi sebelum sampai. Tiada bayaran diperlukan.",

    "Selamat pagi dari AIA Malaysia mengenai polisi insurans hayat anda yang berakhir {digits}. Premium {amount} akan dibayar bulan depan pada 15hb. Hanya mengesahkan butiran perhubungan anda. Anda boleh urus polisi melalui aplikasi AIA.",

    "{bank} Premier. Hanya peringatan anda ada temujanji dengan pengurus hubungan Cik {name} di cawangan {city} esok pukul 2 petang. Sila bawa IC. Untuk menjadual semula, hubungi {hotline}.",

    "{bank} mengesahkan pemindahan dana anda sebanyak {amount} dari akaun anda yang berakhir {digits}. Jika anda tidak membenarkan ini, sila hubungi {hotline} segera.",
]

# Mixed (Manglish) templates — code-switched
HIGH_RISK_TEMPLATES_MIXED = [
    "Hello, ini {bank} security department. We detected unauthorised access pada {bank2u} account you. Please share full IC number dan online banking password sekarang juga supaya kami secure your account.",
    "Good afternoon, calling dari {bank} fraud team. Your account at risk. We need you share OTP yang baru hantar ke phone untuk verify identity.",
    "Saya from {bank} compliance department. Untuk prevent your account from being frozen, please share your Secure2u approval code right now.",
    "This is {bank}. Kami transferring you to fraud officer. To proceed, please provide full {bank} card number and CVV code on the back.",

    "Hello, this is {bank} fraud monitoring. Your account dah compromised. For your safety, please transfer all funds ke our secure holding account: {account}.",
    "Calling dari {bank}. Suspicious activity detected. Please move savings ke this temporary protected account: {account}. Don't tell anyone pasal call ni.",
    "{bank} security alert. Your money at risk. Transfer immediately ke {account} for safekeeping while we investigate.",

    "This is Sergeant {name} dari {agency}. Your IC dah digunakan for money laundering. Ada warrant for your arrest. Pay {amount} bail ke {account} immediately atau officers will come to your house.",
    "Inspector {name} dari PDRM. Your bank account linked to drug trafficking. To clear your name, transfer {amount} ke {account} sekarang.",
    "Ini {agency}. You being investigated for criminal activity. To avoid arrest, you mesti pay bail {amount} immediately.",

    "Important call dari LHDN. You ada outstanding tax {amount}. Pay dalam 24 hours atau warrant will be issued. Press 1 untuk speak dengan our enforcement officer now.",
    "LHDN audit department. Your business ada unpaid taxes {amount}. To avoid asset seizure, transfer payment immediately ke {account}.",

    "Congratulations! You dah selected for exclusive forex programme guaranteeing 30 percent monthly returns. Just deposit {amount} ke {account} untuk secure your spot. This offer expires today.",
    "I'm senior crypto analyst. We ada insider info on token yang akan 10x this week. Minimum investment {amount}. Wallet address coming on WhatsApp.",
    "Hi, ini {name} dari private investment fund. Members made {amount} last month. Guaranteed returns, no risk. Send {amount} ke {account} untuk start.",

    "Good news! Your personal loan {amount} dah pre-approved. To release funds today, just pay processing fee RM 500 ke {account}. Limited time offer.",
    "Your loan application approved. Transfer insurance fee RM 800 ke {account} dalam sejam atau the approval akan dibatalkan.",

    "Hello, ini {telco}. Your line dah flagged for illegal activities and akan terminated dalam 2 hours. Provide IC dan pay security deposit {amount} ke {account} untuk prevent suspension.",
    "{telco} customer service. Your number digunakan for criminal activity. Verify identity by sharing IC, MyKad number, dan bank details now.",

    "Hello, ini {delivery}. A parcel under your name dah intercepted at customs containing illegal items. I'm transferring you ke investigating officer. You need pay fine {amount} ke {account} untuk clear your name.",
    "{delivery} customs department. Your parcel ada contraband. To avoid criminal charges, transfer inspection fee {amount} ke {account} immediately.",

    "Baby, I'm stuck at airport in London dan my wallet kena curi. I need {amount} urgently for ticket home. Please transfer ke {account}. I promise I'll pay back. Don't tell anyone, I'm so embarrassed.",
    "Honey, my mother dekat hospital and I tak boleh access account. Can you transfer {amount} ke {account} for medical bills? I love you and akan bayar balik next week.",

    "This is {bank} digital banking. Your {app} app needs update. Click link saya hantar ke SMS untuk download secure update. You'll need re-enter username dan password untuk verify.",
    "{bank} security update. Install new app dari link saya baru hantar. Then share verification code yang muncul on your screen.",

    "I'm from {bank} technical support. To fix your account issue, please download TeamViewer dan give me access code. We need remotely check your phone.",
    "{bank} IT department. Install AnyDesk now supaya we boleh resolve security alert on your device. Share 9-digit code yang muncul.",

    "Good morning, Bogus Bank security team. Suspicious activity pada account you. I need full username dan password untuk verify identity. Ini standard procedure.",
    "Bogus Bank calling. Please share 6-digit OTP and install TeamViewer supaya kami boleh secure account. Jangan beritahu sesiapa pasal call ni.",
]

MEDIUM_RISK_TEMPLATES_MIXED = [
    "Ini {company} Credit. Your credit card payment {amount} overdue 45 days. If payment tak diterima by Friday, kami akan take legal action and report ke CTOS. Commit to a date now.",
    "Hi, saya dari insurans {company}. Your current coverage sangat inadequate. Kalau anything happens, family you akan dalam trouble. I ada plan for {amount} per month. I need answer today, the rate expires tonight.",
    "{telco} retention department. I see you nak port nombor out. Stay with us — I can offer 50 percent off untuk 6 months, tapi I need MyKad number you to process this sekarang.",

    "Hello, ini {bank}. We noticed unusual login attempts dari new device. For security, please call us back at {hotline} dalam 30 minutes. Have account number dan IC ready.",
    "{bank} fraud team. Suspicious activity detected. Please call us back urgently at {hotline}. Don't delay atau account anda mungkin locked.",

    "I'm conducting financial wellness survey for Bank Negara. I need monthly income, total savings, number of accounts, dan outstanding loans. Can I also get full name dan IC number you?",
    "Government health survey. Kami perlukan full name, IC, address, dan household income for national statistics. Survey ambil 10 minutes.",

    "Good evening, calling dari Malaysian Red Crescent fundraiser for flood relief kat Kelantan. Would you like donate? I can give account number sekarang. For donations {amount} or more, kami akan hantar tax-exempt receipt.",
    "Hi, ini Children's Aid Foundation. We're raising funds untuk orphans. Boleh anda donate {amount} ke account: {account}? Setiap ringgit helps.",

    "I'm calling pasal special timeshare opportunity. Limited slots, exclusive offer, tapi I need decision you today. Can I get IC dan bank details to reserve?",
    "Hi, you've been selected untuk luxury holiday package worth {amount}. Just pay booking fee RM 500 to confirm. Offer expires 6pm today.",

    "Ini third notice dari {company} Credit Recovery. Your debt {amount} mesti settled by Monday atau kami proceed to court. Apa your payment plan?",
    "Final warning. Unpaid bill {amount} akan referred ke collections agency tomorrow. Make payment sekarang to avoid legal action.",
]

LOW_RISK_TEMPLATES_MIXED = [
    "Hi, calling dari {bank} card centre. We have exclusive zero percent instalment plan untuk 12 months on purchases above {amount} at selected merchants. You nak hear more?",
    "{telco} customer service. Kami ada new plan with 100GB data and unlimited calls untuk RM 80 monthly, RM 20 less than your current plan. No commitment. Interested?",
    "{bank} Premier service. We'd like offer you a complimentary financial planning session at flagship branch. Are you free next week?",

    "Calling dari {bank} customer experience team. You recently visited {city} branch. We'd appreciate 2-minute feedback survey. Kami tak akan ask personal details. Skip any question yang you tak comfortable.",
    "Hi, ini quick feedback call dari {company}. How would you rate recent experience with us pada scale 1 to 5?",

    "Reminder: appointment dengan {bank} relationship manager pukul 2pm tomorrow at {city} branch. Please bring IC. To reschedule, call {hotline}.",
    "Hello, ini your insurance agent. Just reminder annual policy review next week. No payment needed at this stage.",
]

SAFE_TEMPLATES_MIXED = [
    "Hello, calling dari {bank} regarding transaction on card ending {digits}. Ada charge {amount} at {merchant} kat {city} today. Was this you? Kalau tak, we'll block the card and you boleh visit any branch with IC.",
    "{bank} fraud monitoring. Card ending {digits} digunakan for {amount} at {merchant} pukul 3pm. Did you make this purchase? Kalau tak, kami akan freeze the card immediately.",

    "Hello, courtesy call dari {bank}. Home loan instalment {amount} due on 20th. Kalau dah bayar, please disregard. For questions call {hotline} atau visit any branch.",
    "{bank} reminder: credit card minimum payment {amount} due dalam 3 days. Payment boleh dibuat via {bank2u} atau any branch.",

    "{bank} calling pasal recent application. Could you confirm last 4 digits IC for verification? Kami tak perlukan other personal information.",
    "Hello dari {bank}. To process card replacement request, please confirm name and last 4 digits account. Thank you.",

    "Hello, ini {delivery}. We have parcel for you, tracking {tracking}. Kami cuba deliver today tapi tiada orang at home. You nak redelivery tomorrow 9am to 12pm, atau pickup dari {city} hub?",
    "{delivery} dispatch. Your package out for delivery today. Driver akan call before arrival. No payment needed.",

    "Good morning dari AIA Malaysia regarding life insurance policy ending {digits}. Premium {amount} due next month on 15th. Just confirming contact details. You boleh manage policy through AIA app.",

    "{bank} Premier. Just reminder you ada appointment dengan relationship manager Ms {name} at {city} branch tomorrow 2pm. Please bring IC. To reschedule, call {hotline}.",

    "{bank} confirming fund transfer {amount} dari account ending {digits}. Kalau you tak authorize this, please call {hotline} immediately.",
]

# Mandarin templates
HIGH_RISK_TEMPLATES_ZH = [
    "您好,这里是{bank}安全部门。我们检测到您的{bank2u}账户有未经授权的访问。请立即提供您的完整身份证号码和网上银行密码,以便我们保护您的账户。",
    "下午好,这里是{bank}反欺诈团队。您的账户面临风险。我们需要您提供刚刚发送到您手机上的OTP验证码以确认身份。",
    "我是{bank}合规部的。为了防止您的账户被冻结,请立即提供您的Secure2u批准代码。",
    "这里是{bank}。我们将把您转接到反欺诈官员。请提供您完整的{bank}卡号和卡背面的CVV代码以继续。",

    "您好,这里是{bank}欺诈监控部。您的账户已被入侵。为了您的安全,请将所有资金转入我们的安全保管账户:{account}。",
    "{bank}打来电话。检测到可疑活动。请将您的存款转移到这个临时保护账户:{account}。不要告诉任何人这通电话。",
    "{bank}安全警报。您的钱面临风险。请立即转入{account}保管,我们正在调查。",

    "我是{agency}的{name}警长。您的身份证被用于洗钱活动。已对您发出逮捕令。请立即将{amount}保释金转入{account},否则警员将上门。",
    "我是PDRM的{name}督察。您的银行账户与毒品贩运有关。要洗清您的名声,请立即转账{amount}到{account}。",
    "这里是{agency}。您正在因犯罪活动被调查。要避免被捕,您必须立即支付{amount}保释金。",

    "重要电话,LHDN内陆税务局。您有{amount}的拖欠税款。请在24小时内付款,否则将发出逮捕令。请按1与我们的执法官员通话。",
    "LHDN审计部。您的企业有{amount}未付税款。要避免资产被查封,请立即将款项转入{account}。",

    "恭喜!您被选中参加独家外汇计划,保证每月30%回报。只需将{amount}存入{account}即可保留您的位置。优惠今天到期。",
    "我是高级加密分析师。我们有内幕消息一个代币本周将上涨10倍。最低投资{amount}。钱包地址将通过WhatsApp发送。",
    "您好,我是{name},来自一家私人投资基金。会员上个月赚了{amount}。保证回报,零风险。发送{amount}到{account}开始。",

    "好消息!您的{amount}个人贷款已预批准。要今天放款,只需支付RM 500手续费到{account}。限时优惠。",
    "您的贷款申请获批。请在一小时内将RM 800保险费转入{account},否则批准将被取消。",

    "您好,这里是{telco}。您的电话线已被标记为非法活动,将在2小时内终止。提供您的身份证并将{amount}保证金转入{account}以防止暂停。",
    "{telco}客户服务。您的号码被用于犯罪活动。请立即提供身份证、MyKad号码和银行详情验证身份。",

    "您好,这里是{delivery}。一个以您名字的包裹在海关被截获,内含违禁品。我将把您转给调查官。您需要支付{amount}罚款到{account}以洗清您的名字。",
    "{delivery}海关部门。您的包裹含有违禁品。要避免刑事指控,请立即将{amount}检查费转入{account}。",

    "宝贝,我被困在伦敦机场,钱包被偷了。我急需{amount}买票回家。请转账到{account}。我保证会还你。不要告诉任何人,我太尴尬了。",
    "亲爱的,我妈妈在医院,我无法访问账户。你能转{amount}到{account}帮忙付医药费吗?我爱你,下周还你。",

    "这里是{bank}数字银行支持。您的{app}应用需要更新。点击我发送到您短信的链接下载安全更新。您需要重新输入用户名和密码进行验证。",
    "{bank}安全更新。从我刚发送的链接安装新应用。然后分享屏幕上出现的验证码。",

    "我是{bank}技术支持。要修复您的账户问题,请下载TeamViewer并给我访问代码。我们需要远程检查您的手机。",
    "{bank} IT部门。立即安装AnyDesk,以便我们解决您设备上的安全警报。分享出现的9位代码。",

    "早上好,Bogus Bank安全团队。您的账户有可疑活动。我需要您完整的用户名和密码来验证身份。这是标准程序。",
    "Bogus Bank来电。请分享您的6位OTP并安装TeamViewer,以便我们保护您的账户。不要告诉任何人这通电话。",
]

MEDIUM_RISK_TEMPLATES_ZH = [
    "这里是{company}信贷。您的{amount}信用卡欠款已逾期45天。如果周五前未收到付款,我们将采取法律行动并报告CTOS。请现在承诺日期。",
    "您好,我是{company}保险的。您当前的保障非常不足。如果发生意外,您的家人将陷入麻烦。我有一个每月{amount}的计划。我今天需要您的答复,优惠今晚到期。",
    "{telco}挽留部门。我看到您想转出号码。留在我们这里,我可以提供6个月50%折扣,但我现在需要您的MyKad号码处理。",

    "您好,这里是{bank}。我们注意到来自新设备的异常登录尝试。出于安全考虑,请在30分钟内回拨{hotline}。请准备好账号和身份证。",
    "{bank}反欺诈团队。检测到可疑活动。请紧急回拨{hotline}。不要拖延,否则您的账户可能被锁定。",

    "我正在为国家银行进行金融健康调查。我需要您的月收入、总储蓄、账户数量和未偿还贷款。我也可以获取您的全名和身份证号码吗?",
    "政府卫生调查。我们需要您的全名、身份证、地址和家庭收入用于国家统计。调查需要10分钟。",

    "晚上好,这里是马来西亚红新月会的吉兰丹水灾救援募捐。您愿意捐款吗?我现在可以给您我们的账号。{amount}或以上的捐款,我们会发送免税收据。",
    "您好,这里是儿童援助基金会。我们正在为孤儿筹款。您能捐{amount}到我们的账户吗:{account}?每一令吉都有帮助。",

    "我打电话是关于一个特别的分时度假机会。名额有限,独家优惠,但我今天需要您的决定。我可以获取您的身份证和银行详情来预订吗?",
    "您好,您被选中获得价值{amount}的豪华假期套餐。只需支付RM 500预订费即可确认。优惠今天下午6点到期。",

    "这是{company}信贷追讨的第三次通知。您的{amount}债务必须在周一前结清,否则我们将提交法庭。您的付款计划是什么?",
    "最后警告。您未付的{amount}账单明天将转交收账机构。请立即付款以避免法律行动。",
]

LOW_RISK_TEMPLATES_ZH = [
    "您好,这里是{bank}信用卡中心。我们有一个独家12个月零利率分期付款计划,适用于在指定商户的{amount}以上购物。您想了解更多吗?",
    "{telco}客户服务。我们有新计划,每月RM 80提供100GB数据和无限通话,比您当前的计划便宜RM 20。无合约。感兴趣吗?",
    "{bank}尊贵服务。我们想为您提供旗舰分行的免费财务规划课程。下周您方便吗?",

    "{bank}客户体验团队来电。您最近访问了我们的{city}分行。我们希望您参加2分钟的反馈调查。我们不会询问个人详情。您可以跳过任何不舒服的问题。",
    "您好,这是来自{company}的快速反馈电话。在1到5的范围内,您如何评价最近与我们的体验?",

    "提醒:您与{bank}客户经理的预约是明天下午2点在{city}分行。请带上您的身份证。要重新安排,请致电{hotline}。",
    "您好,我是您的保险代理。只是提醒您下周年度保单审查。此时无需付款。",
]

SAFE_TEMPLATES_ZH = [
    "您好,这里是{bank},关于您卡号尾号{digits}的交易。今天在{city}的{merchant}有一笔{amount}的消费。是您本人吗?如果不是,我们会冻结卡片,您可以带身份证到任何分行。",
    "{bank}欺诈监控。您尾号{digits}的卡片下午3点在{merchant}消费{amount}。是您本人购买的吗?如果不是,我们将立即冻结卡片。",

    "您好,这是{bank}的礼貌电话。您的房贷分期{amount}将于20日到期。如果您已付款,请忽略。如有疑问请致电{hotline}或访问任何分行。",
    "{bank}提醒:您的信用卡最低还款额{amount}将在3天内到期。可通过{bank2u}或任何分行付款。",

    "{bank}就您最近的申请来电。您能确认身份证最后4位数字以供验证吗?我们不需要其他个人信息。",
    "您好,{bank}。要处理您的换卡请求,请确认您的姓名和账户最后4位数字。谢谢。",

    "您好,这里是{delivery}。我们有您的包裹,追踪号{tracking}。我们今天尝试派送但无人在家。您希望明天上午9点到12点重送,还是从我们的{city}枢纽自取?",
    "{delivery}调度。您的包裹今天派送中。司机会在到达前致电。无需付款。",

    "早上好,AIA马来西亚就您尾号{digits}的人寿保单来电。您的{amount}保费将于下个月15日到期。只是确认您的联系详情。您可以通过AIA应用管理保单。",

    "{bank}尊贵服务。提醒您明天下午2点与{city}分行的{name}女士有预约。请带身份证。要重新安排,请致电{hotline}。",

    "{bank}确认您从尾号{digits}账户的{amount}转账。如果您未授权此交易,请立即致电{hotline}。",
]


# ── Generator ──────────────────────────────────────────────────────────────

def fill_template(template: str, lang: str, rng: random.Random) -> str:
    """Replace slots in a template with random language-appropriate values."""
    bank = rng.choice(BANKS)
    bank_name = bank["name_zh"] if lang == "zh" else bank[f"name_{lang.split('_')[0]}" if lang in ("en", "ms") else "name_en"]
    if lang == "ms": bank_name = bank["name_ms"]
    elif lang == "en" or lang == "mixed": bank_name = bank["name_en"]
    elif lang == "zh": bank_name = bank["name_zh"]

    bank2u_map = {
        "Maybank": "Maybank2u", "Public Bank": "PBe", "CIMB": "CIMB Clicks",
        "RHB Bank": "RHB Now", "Hong Leong": "HLB Connect", "AmBank": "AmOnline",
        "Bank Islam": "GO", "BSN": "myBSN",
    }
    bank2u = bank2u_map.get(bank["name_en"], "online banking")
    if lang == "zh":
        bank2u = "网上银行"
    elif lang == "ms":
        bank2u_ms = {"Maybank": "Maybank2u", "Public Bank": "PBe", "CIMB": "CIMB Clicks"}
        bank2u = bank2u_ms.get(bank["name_en"], "perbankan dalam talian")

    app_map = {
        "Maybank": "MAE", "Public Bank": "PB engage", "CIMB": "CIMB OCTO",
        "RHB Bank": "RHB Mobile", "Hong Leong": "HLB Connect", "AmBank": "AmOnline",
        "Bank Islam": "GO", "BSN": "myBSN",
    }
    app = app_map.get(bank["name_en"], "mobile banking")

    merchants = {"en": MERCHANTS_EN, "ms": MERCHANTS_MS, "zh": MERCHANTS_ZH, "mixed": MERCHANTS_EN}[lang]
    cities = {"en": CITIES_EN, "ms": CITIES_MS, "zh": CITIES_ZH, "mixed": CITIES_EN}[lang]

    company_options = ["AEON", "RCE Capital", "Aspire", "Asia Brilliant"]
    company_zh = ["永旺", "RCE金融", "Aspire", "亚洲光辉"]

    return template.format(
        bank=bank_name,
        bank2u=bank2u,
        app=app,
        amount=rng.choice(AMOUNTS),
        digits=rng.choice(CARD_DIGITS),
        merchant=rng.choice(merchants),
        city=rng.choice(cities),
        account=rng.choice(ACCOUNT_NUMBERS),
        agency=rng.choice([a["name"] for a in GOV_AGENCIES_EN]),
        name=rng.choice(NAMES_EN),
        telco=rng.choice(TELCOS),
        delivery=rng.choice(DELIVERY_COMPANIES),
        tracking=f"{rng.choice(['JT', 'PL', 'LZ'])}{rng.randint(20240000, 20269999)}",
        company=rng.choice(company_zh if lang == "zh" else company_options),
        hotline=bank["hotline"],
    )


def generate_for_language(lang: str, count: int = 100) -> list[dict]:
    """Generate `count` scenarios for a language. Balanced across risk levels."""
    templates_by_risk = {
        "en": {
            "HIGH_RISK": HIGH_RISK_TEMPLATES_EN,
            "MEDIUM_RISK": MEDIUM_RISK_TEMPLATES_EN,
            "LOW_RISK": LOW_RISK_TEMPLATES_EN,
            "SAFE": SAFE_TEMPLATES_EN,
        },
        "ms": {
            "HIGH_RISK": HIGH_RISK_TEMPLATES_MS,
            "MEDIUM_RISK": MEDIUM_RISK_TEMPLATES_MS,
            "LOW_RISK": LOW_RISK_TEMPLATES_MS,
            "SAFE": SAFE_TEMPLATES_MS,
        },
        "mixed": {
            "HIGH_RISK": HIGH_RISK_TEMPLATES_MIXED,
            "MEDIUM_RISK": MEDIUM_RISK_TEMPLATES_MIXED,
            "LOW_RISK": LOW_RISK_TEMPLATES_MIXED,
            "SAFE": SAFE_TEMPLATES_MIXED,
        },
        "zh": {
            "HIGH_RISK": HIGH_RISK_TEMPLATES_ZH,
            "MEDIUM_RISK": MEDIUM_RISK_TEMPLATES_ZH,
            "LOW_RISK": LOW_RISK_TEMPLATES_ZH,
            "SAFE": SAFE_TEMPLATES_ZH,
        },
    }[lang]

    # Distribution: 40% HIGH_RISK, 25% MEDIUM_RISK, 15% LOW_RISK, 20% SAFE
    target_counts = {
        "HIGH_RISK": int(count * 0.40),
        "MEDIUM_RISK": int(count * 0.25),
        "LOW_RISK": int(count * 0.15),
        "SAFE": count - int(count * 0.40) - int(count * 0.25) - int(count * 0.15),
    }

    rng = random.Random(hash(lang) & 0xFFFFFFFF)
    out = []
    for risk_level, n in target_counts.items():
        templates = templates_by_risk[risk_level]
        # Cycle through templates with different fillers to get N unique
        for i in range(n):
            tmpl = templates[i % len(templates)]
            text = fill_template(tmpl, lang, rng)
            out.append({
                "name": f"{lang}_{risk_level.lower()}_{i+1:03d}",
                "text": text,
                "risk_level": risk_level,
                "lang": lang,
            })

    rng.shuffle(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="all",
                        help="en | ms | mixed | zh | all")
    parser.add_argument("--count", type=int, default=100,
                        help="Scenarios per language")
    parser.add_argument("--output-dir", default=str(OUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = ["en", "ms", "mixed", "zh"] if args.lang == "all" else [args.lang]

    for lang in langs:
        scenarios = generate_for_language(lang, args.count)
        out_path = out_dir / f"{lang}.json"
        out_path.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False))

        # Distribution summary
        from collections import Counter
        dist = Counter(s["risk_level"] for s in scenarios)
        print(f"  {lang}: {len(scenarios)} scenarios → {out_path}")
        for k in ("HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "SAFE"):
            print(f"    {k:<14} {dist.get(k, 0)}")

    print(f"\nDone. Files written to {out_dir}/")
    print("Next: generate audio with `uv run tools/tts/generate.py` (needs Mandarin TTS support)")


if __name__ == "__main__":
    main()
