import meerkat as mk

network = mk.gui.start()

original_data = mk.DataFrame(
    {
        "InvoiceNo": [43, 34, 43, 65, 50] * 20,
        "UnitPrice": [403, 304, 23, 23, 193] * 20,
        "UserId": [0, 1, 2, 3, 4] * 20,
    }
)

table = mk.gui.Table(df=original_data)

mk.gui.Interface(component=table).launch()
