//+------------------------------------------------------------------+
//| AltrixTrend.mq5                                                  |
//| Trend lent (ansamblu L120/L250) cu vol targeting, portofoliu     |
//| multi-simbol, rebalansare saptamanala. Garduri FTMO incluse.     |
//| Portat din laboratorul Altrix (studiile 11/12). Se ataseaza pe   |
//| ORICE grafic (ex. EURUSD D1); gestioneaza toate simbolurile.     |
//+------------------------------------------------------------------+
#property copyright "Altrix"
#property version   "1.01"
#property description "Trend following zilnic, vol targeting, garduri FTMO (zilnic 3.5%, total 8.5%)."
#include <Trade/Trade.mqh>

input string InpSymbols        = "EURUSD,GBPUSD,USDJPY,XAUUSD,US100.cash,US500.cash,US30.cash,GER40.cash,UK100.cash,JP225.cash"; // simboluri (numele din MT5-ul tau)
input string InpLookbacks      = "120,250";   // zile; semnal = media semnelor
input double InpTargetVolPct   = 10.0;        // vol anuala tinta per simbol (%)
input double InpMaxLeverage    = 4.0;         // plafon levier per simbol (x equity, inainte de impartirea la nr. simboluri active)
input int    InpVolSpan        = 60;          // EWMA vol (zile)
input double InpRebalBand      = 0.25;        // nu retranzactiona daca volumul tinta difera < 25% de cel curent
input double InpDailyStopPct   = 3.5;         // hard-stop zilnic (% din max(balance,equity) la inceputul zilei server)
input double InpMaxDDPct       = 8.5;         // oprire totala (% din balanta initiala). FTMO: 10%
input double InpInitialBalance = 0;           // 0 = balanta la prima pornire; pune dimensiunea contului FTMO daca ai pornit tarziu
input double InpProfitTargetPct= 0;           // 0 = oprit; ex. 10 / 5 = inchide tot si sta pe loc la tinta
input double InpEmergencySL_ATR= 6.0;         // stop de urgenta la N x ATR(20) zilnic; 0 = fara stop
input long   InpMagic          = 20261005;
input string InpNtfyTopic      = "";          // topic ntfy.sh (gol = fara notificari); permite https://ntfy.sh in Tools>Options>Expert Advisors
input bool   InpTradeEnabled   = true;        // false = doar calculeaza si afiseaza (mod uscat)

CTrade   trade;
string   g_sym[];  int g_nS = 0;
int      g_lb[];   int g_nL = 0;
int      g_maxLb = 0;
double   g_initBal = 0, g_dayRef = 0;
long     g_dayNo = -1;
bool     g_dayLocked = false;
string   g_gvKill, g_gvInit, g_gvWk;
string   g_info = "";
int      g_fail = 0;            // ordine esuate in rebalansarea curenta (piata inchisa etc.)
int      g_tries = 0;

//+------------------------------------------------------------------+
void Notify(const string msg)
{
   Print("[AltrixTrend] ", msg);
   if(InpNtfyTopic == "") return;
   char data[], res[]; string hdr = "Content-Type: text/plain; charset=utf-8\r\n", rh;
   int n = StringToCharArray("AltrixTrend: " + msg, data, 0, WHOLE_ARRAY, CP_UTF8);
   if(n > 0) ArrayResize(data, n - 1);
   int rc = WebRequest("POST", "https://ntfy.sh/" + InpNtfyTopic, hdr, 5000, data, res, rh);
   if(rc == -1) Print("ntfy: WebRequest a esuat (", GetLastError(), "). Permite URL-ul in Tools > Options > Expert Advisors.");
}

//+------------------------------------------------------------------+
int OnInit()
{
   string parts[];
   int n = StringSplit(InpSymbols, ',', parts);
   ArrayResize(g_sym, 0);
   for(int i = 0; i < n; i++)
   {
      string s = parts[i]; StringTrimLeft(s); StringTrimRight(s);
      if(s == "") continue;
      if(!SymbolSelect(s, true)) { Print("Simbol indisponibil, sar peste: ", s); continue; }
      int k = ArraySize(g_sym); ArrayResize(g_sym, k + 1); g_sym[k] = s;
   }
   g_nS = ArraySize(g_sym);
   if(g_nS == 0) { Print("Niciun simbol valid."); return INIT_FAILED; }

   n = StringSplit(InpLookbacks, ',', parts);
   ArrayResize(g_lb, 0);
   for(int i = 0; i < n; i++)
   {
      int L = (int)StringToInteger(parts[i]);
      if(L < 5) continue;
      int k = ArraySize(g_lb); ArrayResize(g_lb, k + 1); g_lb[k] = L;
      if(L > g_maxLb) g_maxLb = L;
   }
   g_nL = ArraySize(g_lb);
   if(g_nL == 0) { Print("Lookback-uri invalide."); return INIT_FAILED; }

   long acc = AccountInfoInteger(ACCOUNT_LOGIN);
   g_gvKill = "ALTRIX_KILL_" + (string)acc; g_gvInit = "ALTRIX_INIT_" + (string)acc; g_gvWk = "ALTRIX_WK_" + (string)acc;
   if(MQLInfoInteger(MQL_TESTER)) { GlobalVariableDel(g_gvKill); GlobalVariableDel(g_gvInit); GlobalVariableDel(g_gvWk); }

   if(InpInitialBalance > 0) g_initBal = InpInitialBalance;
   else if(GlobalVariableCheck(g_gvInit)) g_initBal = GlobalVariableGet(g_gvInit);
   else g_initBal = AccountInfoDouble(ACCOUNT_BALANCE);
   GlobalVariableSet(g_gvInit, g_initBal);

   trade.SetExpertMagicNumber((ulong)InpMagic);
   trade.SetDeviationInPoints(50);
   for(int i = 0; i < g_nS; i++)                          // diagnostic: cate bare D1 are fiecare simbol (forteaza si sincronizarea istoricului)
      Print("[AltrixTrend] ", g_sym[i], ": bare D1 disponibile = ", Bars(g_sym[i], PERIOD_D1), ", mod tranzactionare = ", (int)SymbolInfoInteger(g_sym[i], SYMBOL_TRADE_MODE));
   EventSetTimer(MQLInfoInteger(MQL_TESTER) ? 3600 : 20);   // in tester ruleaza din ora in ora (suficient pentru un EA saptamanal)
   Notify(StringFormat("pornit: %d simboluri, balanta initiala %.2f, mod %s", g_nS, g_initBal, InpTradeEnabled ? "LIVE" : "USCAT"));
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); Comment(""); }

void OnTick() { /* totul ruleaza din OnTimer, ca sa mearga pe orice grafic */ }

//+------------------------------------------------------------------+
//| pozitii                                                           |
//+------------------------------------------------------------------+
double NetVolume(const string s)
{
   double v = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i); if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != s || PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      double vol = PositionGetDouble(POSITION_VOLUME);
      v += (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? vol : -vol;
   }
   return v;
}

int CountPositions()
{
   int c = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i); if(t == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) == InpMagic) c++;
   }
   return c;
}

void CloseSymbol(const string s)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i); if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != s || PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(!trade.PositionClose(t)) { g_fail++; Print("Inchidere esuata ", s, " rc=", trade.ResultRetcode()); }
   }
}

void CloseAll()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i); if(t == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(!trade.PositionClose(t)) Print("Inchidere esuata ticket ", t, " rc=", trade.ResultRetcode());
   }
}

double DailyATR(const string s)
{
   MqlRates r[];
   ArraySetAsSeries(r, true);
   if(CopyRates(s, PERIOD_D1, 1, 22, r) < 21) return 0;
   double sum = 0;
   for(int k = 0; k < 20; k++)
   {
      double tr = MathMax(r[k].high - r[k].low, MathMax(MathAbs(r[k].high - r[k + 1].close), MathAbs(r[k].low - r[k + 1].close)));
      sum += tr;
   }
   return sum / 20.0;
}

void OpenPos(const string s, const bool buy, const double vol)
{
   if(vol <= 0) return;
   trade.SetTypeFillingBySymbol(s);
   double sl = 0;
   if(InpEmergencySL_ATR > 0)
   {
      double atr = DailyATR(s);
      double px = buy ? SymbolInfoDouble(s, SYMBOL_ASK) : SymbolInfoDouble(s, SYMBOL_BID);
      if(atr > 0 && px > 0)
      {
         double d = InpEmergencySL_ATR * atr;
         int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
         sl = NormalizeDouble(buy ? px - d : px + d, digits);
         if(sl <= 0) sl = 0;
      }
   }
   bool ok = buy ? trade.Buy(vol, s, 0, sl, 0, "ALTRIX") : trade.Sell(vol, s, 0, sl, 0, "ALTRIX");
   if(!ok && sl != 0)                                   // stop invalid pentru broker: reincearca fara stop
      ok = buy ? trade.Buy(vol, s, 0, 0, 0, "ALTRIX") : trade.Sell(vol, s, 0, 0, 0, "ALTRIX");
   if(!ok) { g_fail++; Print("Deschidere esuata ", s, " vol=", vol, " rc=", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription()); }
}

void ReduceVolume(const string s, const bool wasLong, double vol)
{
   bool hedging = ((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE) == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
   trade.SetTypeFillingBySymbol(s);
   if(!hedging)                                         // netting: o tranzactie opusa reduce pozitia
   {
      if(wasLong) trade.Sell(vol, s, 0, 0, 0, "ALTRIX"); else trade.Buy(vol, s, 0, 0, 0, "ALTRIX");
      return;
   }
   double step = SymbolInfoDouble(s, SYMBOL_VOLUME_STEP);
   for(int i = PositionsTotal() - 1; i >= 0 && vol > 1e-9; i--)
   {
      ulong t = PositionGetTicket(i); if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != s || PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      double pv = PositionGetDouble(POSITION_VOLUME);
      double cv = MathMin(vol, pv);
      cv = MathFloor(cv / step + 1e-9) * step;
      if(cv <= 0) continue;
      bool ok = (cv >= pv - 1e-9) ? trade.PositionClose(t) : trade.PositionClosePartial(t, cv);
      if(ok) vol -= cv; else { g_fail++; Print("Reducere esuata ", s, " rc=", trade.ResultRetcode()); }
   }
}

//+------------------------------------------------------------------+
//| semnal + volatilitate                                             |
//+------------------------------------------------------------------+
bool Compute(const string s, double &sig, double &sd)
{
   if(SymbolInfoInteger(s, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_FULL) return false;
   double c[];
   ArraySetAsSeries(c, true);
   int want = MathMax(g_maxLb + 2, 450);
   int got = CopyClose(s, PERIOD_D1, 1, want, c);       // c[0] = ultima bara D1 inchisa
   if(got < g_maxLb + 30) return false;
   double sum = 0;
   for(int j = 0; j < g_nL; j++) sum += (c[0] > c[g_lb[j]]) ? 1.0 : -1.0;
   sig = sum / g_nL;
   double lam = 1.0 - 2.0 / (InpVolSpan + 1.0), v = 0; bool first = true;
   for(int k = got - 2; k >= 0; k--)
   {
      if(c[k + 1] <= 0) continue;
      double r = c[k] / c[k + 1] - 1.0;
      if(first) { v = r * r; first = false; } else v = lam * v + (1.0 - lam) * r * r;
   }
   sd = MathSqrt(v);
   return (sd > 0);
}

double NormLots(const string s, double lots)
{
   double step = SymbolInfoDouble(s, SYMBOL_VOLUME_STEP), mn = SymbolInfoDouble(s, SYMBOL_VOLUME_MIN), mx = SymbolInfoDouble(s, SYMBOL_VOLUME_MAX);
   if(step <= 0) return 0;
   lots = MathFloor(lots / step + 1e-9) * step;
   if(lots < mn) return 0;
   return MathMin(lots, mx);
}

double TargetLots(const string s, const double sig, const double sd, const int nAct, const double eq)
{
   double price = SymbolInfoDouble(s, SYMBOL_BID), tv = SymbolInfoDouble(s, SYMBOL_TRADE_TICK_VALUE), ts = SymbolInfoDouble(s, SYMBOL_TRADE_TICK_SIZE);
   if(price <= 0 || tv <= 0 || ts <= 0 || nAct <= 0) return 0;
   double vppu = tv / ts;                               // valoare in moneda contului a unei miscari de 1.0 in pret, per 1 lot
   double dollars = eq * (InpTargetVolPct / 100.0) / MathSqrt(252.0) / nAct;   // volatilitate zilnica tinta alocata simbolului
   double lots = dollars / (sd * price * vppu);
   double cap = InpMaxLeverage * eq / nAct / (price * vppu);
   lots = MathMin(lots, cap) * MathAbs(sig);
   lots = NormLots(s, lots);
   return (sig > 0) ? lots : -lots;
}

void ApplyTarget(const string s, const double tgt)
{
   double cur = NetVolume(s);
   if(MathAbs(tgt) < 1e-9)
   {
      if(MathAbs(cur) > 1e-9) { Print(s, ": inchid (semnal 0 / sub lot minim)"); CloseSymbol(s); }
      return;
   }
   if(MathAbs(cur) < 1e-9) { OpenPos(s, tgt > 0, MathAbs(tgt)); return; }
   if((cur > 0) != (tgt > 0)) { CloseSymbol(s); OpenPos(s, tgt > 0, MathAbs(tgt)); return; }
   double a = MathAbs(cur), b = MathAbs(tgt);
   if(MathAbs(b - a) / a <= InpRebalBand) return;
   if(b > a) OpenPos(s, tgt > 0, NormLots(s, b - a));
   else      ReduceVolume(s, cur > 0, a - b);
}

void Rebalance()
{
   g_fail = 0;
   double sig[], sd[]; bool ok[]; int nAct = 0;
   ArrayResize(sig, g_nS); ArrayResize(sd, g_nS); ArrayResize(ok, g_nS);
   for(int i = 0; i < g_nS; i++) { ok[i] = Compute(g_sym[i], sig[i], sd[i]); if(ok[i]) nAct++; }
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   string rep = "";
   for(int i = 0; i < g_nS; i++)
   {
      double t = ok[i] ? TargetLots(g_sym[i], sig[i], sd[i], nAct, eq) : 0.0;
      rep += StringFormat("%s:%.2f ", g_sym[i], t);
      if(InpTradeEnabled) ApplyTarget(g_sym[i], t);
   }
   if(g_tries <= 1 || g_fail == 0) Notify(StringFormat("rebalansare (%d active, equity %.2f, esecuri %d): %s", nAct, eq, g_fail, rep));
}

//+------------------------------------------------------------------+
//| garduri FTMO                                                      |
//+------------------------------------------------------------------+
bool Guards()
{
   double bal = AccountInfoDouble(ACCOUNT_BALANCE), eq = AccountInfoDouble(ACCOUNT_EQUITY);
   long dayNo = (long)(TimeCurrent() / 86400);          // ziua serverului (la FTMO = ora CE(S)T, ca in regulile lor)
   if(dayNo != g_dayNo) { g_dayNo = dayNo; g_dayRef = MathMax(bal, eq); g_dayLocked = false; }
   if(GlobalVariableCheck(g_gvKill) && GlobalVariableGet(g_gvKill) > 0) { g_info = "OPRIT (kill activ)"; return false; }
   if(eq <= g_initBal * (1.0 - InpMaxDDPct / 100.0))
   {
      CloseAll(); GlobalVariableSet(g_gvKill, 1);
      Notify(StringFormat("KILL: equity %.2f sub limita totala (%.1f%% din %.2f). Tot inchis, EA oprit. Sterge variabila globala %s ca sa repornesti.", eq, InpMaxDDPct, g_initBal, g_gvKill));
      return false;
   }
   if(InpProfitTargetPct > 0 && eq >= g_initBal * (1.0 + InpProfitTargetPct / 100.0))
   {
      CloseAll(); GlobalVariableSet(g_gvKill, 1);
      Notify(StringFormat("TINTA ATINSA: equity %.2f. Tot inchis, EA in pauza.", eq));
      return false;
   }
   if(!g_dayLocked && eq <= g_dayRef * (1.0 - InpDailyStopPct / 100.0))
   {
      CloseAll(); g_dayLocked = true;
      Notify(StringFormat("HARD-STOP ZILNIC: equity %.2f <= %.2f. Tot inchis pana maine.", eq, g_dayRef * (1.0 - InpDailyStopPct / 100.0)));
   }
   if(g_dayLocked) { g_info = "BLOCAT azi (limita zilnica)"; return false; }
   g_info = "activ";
   return true;
}

void OnTimer()
{
   static bool once = false;
   bool ok = Guards();
   if(!once) { once = true; Print("[AltrixTrend] primul ciclu: garduri=", ok, " tradeEnabled=", InpTradeEnabled, " terminalTrade=", TerminalInfoInteger(TERMINAL_TRADE_ALLOWED), " contTrade=", AccountInfoInteger(ACCOUNT_TRADE_ALLOWED), " W1=", TimeToString(iTime(g_sym[0], PERIOD_W1, 0)), " acum=", TimeToString(TimeCurrent())); }
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(ok && InpTradeEnabled && TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) && AccountInfoInteger(ACCOUNT_TRADE_ALLOWED))
   {
      datetime wk = iTime(g_sym[0], PERIOD_W1, 0);
      double last = GlobalVariableCheck(g_gvWk) ? GlobalVariableGet(g_gvWk) : 0;
      if(wk > 0 && (double)wk != last && TimeCurrent() >= wk + 2 * 3600)
      {
         g_tries++;
         Rebalance();
         if(g_fail == 0 || TimeCurrent() > wk + 24 * 3600)     // reincearca la 20 s daca ordinele au esuat (piata inchisa), maxim 24 h
         {
            GlobalVariableSet(g_gvWk, (double)wk); g_tries = 0;
            if(g_fail > 0) Notify("rebalansare incheiata cu esecuri dupa 24h; vezi jurnalul");
         }
      }
   }
   Comment(StringFormat("AltrixTrend\nStare: %s\nEquity: %.2f | Balanta init.: %.2f\nRef. zi: %.2f | Zi: %+.2f%%\nDD total: %.2f%% (limita %.1f%%)\nPozitii: %d",
           g_info, eq, g_initBal, g_dayRef, g_dayRef > 0 ? (eq / g_dayRef - 1.0) * 100.0 : 0.0, g_initBal > 0 ? (1.0 - eq / g_initBal) * 100.0 : 0.0, InpMaxDDPct, CountPositions()));
}
//+------------------------------------------------------------------+
