# AltrixTrend v2.1 (MT5) – trend lent cu vol targeting, garduri FTMO

v2.1: merge pe orice timeframe (recomandat H1), logica din OnTick (live: si din timer), plafon expunere totala `InpMaxTotalLev` (cont 50k, levier 1:30), lista separata `InpTesterSymbols` pentru Strategy Tester.

## Ce face
- Portofoliu pe 10 simboluri (EURUSD, GBPUSD, USDJPY, XAUUSD, US100, US500, US30, GER40, UK100, JP225 – numele din MT5 FTMO, editabile).
- Semnal zilnic: media semnelor pe 120 si 250 de zile (long / short / jumatate). Rebalansare o data pe saptamana (prima ora dupa deschiderea saptamanii + 2h).
- Marime: volatilitate anuala tinta 10% per simbol, impartita la nr. de simboluri active, plafon levier 4x. Nu retranzactioneaza daca volumul tinta difera sub 25%.
- Garduri: hard-stop zilnic 3.5% (referinta = max(balance, equity) la inceputul zilei serverului), oprire totala la 8.5% din balanta initiala, tinta de profit optionala, stop de urgenta 6 x ATR(20) zilnic.
- Notificare ntfy.sh (optional) la pornire, rebalansare, hard-stop, kill.

## Ce NU este
- Nu e un edge dovedit. In laborator (studiile 11-13, 2016-2026): Sharpe net ~0 pana la +0.2, P(trecere FTMO) ~30-35% la vol 10-15%, fara swap. Valoare asteptata pozitiva doar ca "optiune" pe taxa (vezi analiza). Swap-ul real al brokerului poate scadea randamentul.
- Netestat in MetaEditor/Strategy Tester de mine (nu am MT5 aici). Compileaza-l si ruleaza-l pe DEMO inainte de orice cont real.

## Instalare – pas cu pas
1. In MT5: meniu **File > Open Data Folder**. Se deschide un folder; intra in **MQL5 > Experts**.
2. Copiaza aici fisierul **AltrixTrend.mq5**.
3. Deschide-l cu dublu-click (se deschide MetaEditor) si apasa **F7 (Compile)**. In fereastra de jos trebuie sa apara "0 errors". Daca sunt erori, trimite-mi textul lor.
4. Inapoi in MT5: panoul **Market Watch** (Ctrl+M) > click dreapta > **Symbols**... si verifica ca toate simbolurile din lista exista (nume exact). Daca un nume difera, il schimbi in inputul `InpSymbols`.
5. Deschide un grafic **EURUSD, H1**. Din **Navigator** (Ctrl+N) > **Expert Advisors** trage **AltrixTrend** pe grafic.
6. In fereastra care apare, tab **Common**: bifeaza **Allow Algo Trading**. Tab **Inputs**: pune `InpInitialBalance` = marimea contului (ex. 100000) daca nu pornesti de la balanta initiala; pune `InpTradeEnabled = false` pentru primele zile (mod uscat: doar calculeaza si afiseaza).
7. Sus in MT5 apasa butonul **Algo Trading** (sa fie verde).
8. Notificari ntfy (optional): **Tools > Options > Expert Advisors** > bifeaza **Allow WebRequest for listed URL** > adauga `https://ntfy.sh`. In input `InpNtfyTopic` pune topicul tau (acelasi pe care il urmaresti in aplicatia ntfy de pe telefon).
9. Dupa 1-2 zile in mod uscat, compara in jurnalul din tab **Experts** liniile "rebalansare ... EURUSD:+0.35 ..." cu ce te asteptai, apoi pune `InpTradeEnabled = true`.

## Rulare pe DEMO (recomandat in loc de Strategy Tester pe Mac/Wine)
1. Deschide graficul EURUSD H1 pe contul demo FTMO. Trage AltrixTrend pe grafic.
2. Tab **Inputs**: `InpTradeEnabled = false` (mod uscat), `InpInitialBalance = 50000`. Tab **Common**: bifeaza **Allow Algo Trading**.
3. Jos, tab **Experts**: trebuie sa apara `[AltrixTrend] pornit v2.1: N simboluri (LIVE)...`. In coltul graficului apare starea (equity, DD, pozitii).
4. Rebalansarea se face la inceputul saptamanii (luni, dupa 2 ore). Linia `rebalansare (...)` arata volumele tinta pe fiecare simbol. Cu `InpTradeEnabled = false` nu se deschide nimic: doar compari cu ce te asteptai.
5. Dupa 1-2 saptamani: `InpTradeEnabled = true` pe DEMO, urmaresti ordinele reale, spread-ul si swap-ul.

## Strategy Tester (daca functioneaza la tine)
- **View > Strategy Tester**; Expert: AltrixTrend; Symbol: EURUSD; Period: H1; Modelare: **1 minute OHLC** (tick-urile reale sunt foarte lente, mai ales pe Mac/Wine); interval 2023-2025; Deposit 50000; Levier 1:30.
- In tester se folosesc doar simbolurile din `InpTesterSymbols` (implicit EURUSD, GBPUSD, USDJPY, XAUUSD).
- Compara cu laboratorul: Sharpe 0 - 0.2 e asteptat.

## Parametri importanti
| Input | Valoare | Rol |
|---|---|---|
| InpTargetVolPct | 10 | volatilitate anuala tinta per simbol |
| InpDailyStopPct | 3.5 | hard-stop zilnic |
| InpMaxDDPct | 8.5 | oprire totala (FTMO: 10%) |
| InpEmergencySL_ATR | 6 | stop de urgenta; 0 = fara |
| InpTradeEnabled | true/false | false = mod uscat |
| InpMaxTotalLev | 12 | plafon expunere totala (suma notional / equity) |
| InpTesterSymbols | EURUSD,GBPUSD,USDJPY,XAUUSD | simboluri doar in tester |

Dupa un KILL (limita totala sau tinta), EA-ul ramane oprit. Il repornesti stergand variabila globala **ALTRIX_KILL_<cont>**: in MT5 apasa **F3** (Global Variables), selecteaza-o si apasa **Delete**.
