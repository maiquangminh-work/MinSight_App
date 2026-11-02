"""
vnstock compatibility adapter for vnstock 5.x.
Ensures seamless backwards-compatibility with code written for vnstock 3/4 (Vnstock().stock(...)).
"""
try:
    from vnstock import Quote, Company, Market, Trading
except Exception:
    Quote, Company, Market, Trading = None, None, None, None

class VnstockStock:
    def __init__(self, symbol, source='KBS'):
        self.symbol = symbol.upper()
        src = (source or 'KBS').upper()
        if src in ('TCBS', 'SSI'):
            src = 'KBS'
        self.source = src
        self.quote = Quote(symbol=self.symbol, source=self.source) if Quote else None
        try:
            self.company = Company(source=self.source if self.source in ('VCI', 'KBS', 'CAFEF') else 'KBS', symbol=self.symbol) if Company else None
        except Exception:
            self.company = None
        try:
            self.trading = Trading(source=self.source if self.source in ('VCI', 'KBS') else 'KBS', symbol=self.symbol) if Trading else None
        except Exception:
            self.trading = None

class Vnstock:
    def stock(self, symbol, source='KBS'):
        return VnstockStock(symbol, source)

__all__ = ['Vnstock', 'Quote', 'Company', 'Market', 'Trading']
