import re

# =============================================================================
# Limpieza por cadenas - MÉXICO (Versión 2026)
# =============================================================================
# Patrones para detectar cadenas comerciales mexicanas (locales, regionales
# y multinacionales) que NO son canales de venta target.
# =============================================================================

CHAIN_REGEX = [
    # -------------------------------------------------------------------------
    # SUPERMERCADOS / HIPERMERCADOS / CLUBES DE COMPRA
    # -------------------------------------------------------------------------
    r"\bwalmart\b",
    r"\bbodega\s?aurrera\b",
    r"\bmi\s?bodega\b",
    r"\baurrera\s?express\b",
    r"\bsoriana\b",
    r"\bsoriana\s?(?:h[ií]per|s[uú]per|express)\b",
    r"\bchedraui\b",
    r"\bsuper\s?chedraui\b",
    r"\bselecto\s?chedraui\b",
    r"\bla\s?comer\b",
    r"\bcomercial\s?mexicana\b",
    r"\bfresko\b",
    r"\bcity\s?market\b",
    r"\bsumesa\b",
    r"\bheb\b",
    r"\bh-e-b\b",
    r"\bmi\s?tienda\s?del\s?ahorro\b",
    r"\bs[\s\-]?mart\b",
    r"\bcasaley\b",
    r"\bcasa\s?ley\b",
    r"\bcalimax\b",
    r"\balsuper\b",
    r"\bcostco\b",
    r"\bsam'?s\s?club\b",
    r"\bcity\s?club\b",
    r"\bsuperama\b",

    # -------------------------------------------------------------------------
    # TIENDAS DE CONVENIENCIA / MINIMARKETS
    # -------------------------------------------------------------------------
    r"\boxxo\b",
    r"\b7[\s\-]?eleven\b",
    r"\bseven\s?eleven\b",
    r"\btiendas?\s?extra\b",
    r"\bcircle\s?k\b",
    r"\bgomart\b",
    r"\bgo\s?mart\b",
    r"\bkiosko\b",
    r"\bsupercity\b",
    r"\basturiano\b",
    r"\byepas\b",
    r"\bneto\b",
    r"\btiendas?\s?neto\b",
    r"\btiendas?\s?tres\s?b\b",
    r"\btiendas?\s?3b\b",

    # -------------------------------------------------------------------------
    # FARMACIAS
    # -------------------------------------------------------------------------
    r"\bfarmacias?\s?del\s?ahorro\b",
    r"\bfarmacias?\s?guadalajara\b",
    r"\bfarmacias?\s?benavides\b",
    r"\bfarmacias?\s?similares\b",
    r"\bfarmacia\s?san\s?pablo\b",
    r"\bfarmacias?\s?yza\b",
    r"\bfarmacias?\s?roma\b",
    r"\bfarmapronto\b",
    r"\bfarmacias?\s?gi\b",

    # -------------------------------------------------------------------------
    # GASOLINERAS / ESTACIONES DE SERVICIO
    # -------------------------------------------------------------------------
    r"\bpemex\b",
    r"\boxxo\s?gas\b",
    r"\bpetro[\s\-]?7\b",
    r"\bbp\b",
    r"\bshell\b",
    r"\bchevron\b",
    r"\bgulf\b",
    r"\bg500\b",
    r"\bhidrosina\b",
    r"\brepsol\b",
    r"\btotalenergies\b",

    # -------------------------------------------------------------------------
    # FAST FOOD / RESTAURANTES / CAFETERÍAS
    # -------------------------------------------------------------------------
    r"\bsanborns\b",
    r"\bvips\b",
    r"\btoks\b",
    r"\bel\s?port[oó]n\b",
    r"\bla\s?casa\s?de\s?to[ñn]o\b",
    r"\bburger\s?king\b",
    r"\bmc\s?donald'?s?\b",
    r"\bcarl'?s\s?jr\b",
    r"\bkfc\b",
    r"\bkentucky\s?fried\s?chicken\b",
    r"\bsubway\b",
    r"\bdomino'?s\s?pizza\b",
    r"\bpizza\s?hut\b",
    r"\blittle\s?caesars\b",
    r"\bpapa\s?john'?s\b",
    r"\bstarbucks\b",
    r"\bcielito\s?querido\b",
    r"\bitalianni'?s\b",
    r"\bchili'?s\b",
    r"\bapplebee'?s\b",
    r"\bdairy\s?queen\b",
    r"\bnutrisa\b",
    r"\bsanta\s?clara\b",
    r"\bel\s?globo\b",
    r"\bpasteler[ií]as?\s?la\s?esperanza\b",

    # -------------------------------------------------------------------------
    # BANCOS / FINANCIERAS
    # -------------------------------------------------------------------------
    r"\bbbva\b",
    r"\bbancomer\b",
    r"\bbanamex\b",
    r"\bcitibanamex\b",
    r"\bsantander\b",
    r"\bbanorte\b",
    r"\bhsbc\b",
    r"\bscotiabank\b",
    r"\binbursa\b",
    r"\bbanco\s?azteca\b",
    r"\bbancoppel\b",
    r"\bcompartamos\s?banco\b",

    # -------------------------------------------------------------------------
    # RETAIL / DEPARTAMENTALES / ENTRETENIMIENTO
    # -------------------------------------------------------------------------
    r"\bcoppel\b",
    r"\belektra\b",
    r"\bliverpool\b",
    r"\bel\s?palacio\s?de\s?hierro\b",
    r"\bsears\b",
    r"\bsuburbia\b",
    r"\bhome\s?depot\b",
    r"\boffice\s?depot\b",
    r"\bofficemax\b",
    r"\bautozone\b",
    r"\bcin[eé]polis\b",
    r"\bcinemex\b",
    r"\bsmart\s?fit\b",
    r"\bsport\s?city\b",

    # -------------------------------------------------------------------------
    # TELECOMUNICACIONES
    # -------------------------------------------------------------------------
    r"\btelcel\b",
    r"\bat&t\b",
    r"\bmovistar\b",

    # -------------------------------------------------------------------------
    # COMPETIDORES Y GENÉRICOS
    # -------------------------------------------------------------------------
    r"\bpepsi\b",
    r"\blaboratorio\b",
    r"\balbergue\b",
    r"\bcentro\s?de\s?salud\b",
    r"\bcl[ií]nica\b",
    r"\bagro", 
    r"\bvet",
]

chain_re = re.compile(
    "|".join(f"(?:{p})" for p in CHAIN_REGEX),
    flags=re.IGNORECASE,
)