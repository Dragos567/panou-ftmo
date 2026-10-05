//+------------------------------------------------------------------+
//| AltrixPing.mq5 - test minim: scrie in Jurnal la pornire si la    |
//| fiecare bara noua. Nu tranzactioneaza.                            |
//+------------------------------------------------------------------+
#property version "1.00"
datetime g_last = 0;
int OnInit() { Print("[AltrixPing] pornit pe ", _Symbol, " tester=", (bool)MQLInfoInteger(MQL_TESTER)); return INIT_SUCCEEDED; }
void OnTick()
{
   datetime t = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(t != g_last) { g_last = t; Print("[AltrixPing] bara noua ", TimeToString(t)); }
}
