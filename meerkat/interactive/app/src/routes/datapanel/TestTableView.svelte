<script lang="ts">
    import TableView from "./TableView.svelte";

    let columns: Array<string> = ['metadata', 'image', 'attribute', 'd', 'e'];
    let types: Array<string> = ['string', 'image', 'string', 'string', 'string'];
    let rows: Array<any> = [
        ['a1', "https://i.natgeofe.com/k/830b5d15-92db-429f-a80a-cc89b5700af5/mt-everest.jpg?w=636&h=437",'c1', 'a man in orange', 'no wikipedia article found'],
        ['a2', 'https://upload.wikimedia.org/wikipedia/commons/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg','c2', 'a man in orange', 'no wikipedia article found'],
        ['a3', 'https://images.unsplash.com/photo-1585409677983-0f6c41ca9c3b?ixlib=rb-1.2.1&w=1080&fit=max&q=80&fm=jpg&crop=entropy&cs=tinysrgb','c3', 'a man in orange', 'no wikipedia article found'],
        ['a4', 'https://images.immediate.co.uk/production/volatile/sites/4/2021/08/mountains-7ddde89.jpg?quality=90&resize=768,574','c4', 'a man in orange', 'no wikipedia article found'],
        ['a5', 'https://www.qries.com/images/banner_logo.png','c5', 'a man in orange', 'no wikipedia article found'],
        ['a6', 'https://www.qries.com/images/banner_logo.png','c6', 'a man in orange', 'no wikipedia article found'],
        ['a7', 'https://www.qries.com/images/banner_logo.png','c7', 'a man in orange', 'no wikipedia article found'],
        ['a8', 'https://www.qries.com/images/banner_logo.png','c8', 'a man in orange', 'no wikipedia article found'],
        ['a9', 'https://www.qries.com/images/banner_logo.png','c9', 'a man in orange', 'no wikipedia article found'],
        ['a10', 'https://www.qries.com/images/banner_logo.png','c10', 'a man in orange', 'no wikipedia article found'],
        ['a11', 'https://www.qries.com/images/banner_logo.png','c11', 'a man in orange', 'no wikipedia article found'],
        ['a12', 'https://www.qries.com/images/banner_logo.png','c12', 'a man in orange', 'no wikipedia article found'],
        ['a13', 'https://www.qries.com/images/banner_logo.png','c13', 'a man in orange', 'no wikipedia article found'],
        ['a14', 'https://www.qries.com/images/banner_logo.png','c14', 'a man in orange', 'no wikipedia article found'],
        ['a15', 'https://www.qries.com/images/banner_logo.png','c15', 'a man in orange', 'no wikipedia article found'],
        ['a16', 'https://www.qries.com/images/banner_logo.png','c16', 'a man in orange', 'no wikipedia article found'],
        ['a17', 'https://www.qries.com/images/banner_logo.png','c17', 'a man in orange', 'no wikipedia article found'],
        ['a18', 'https://www.qries.com/images/banner_logo.png','c18', 'a man in orange', 'no wikipedia article found'],
        ['a19', 'https://www.qries.com/images/banner_logo.png','c19', 'a man in orange', 'no wikipedia article found'],
        ['a20', 'https://www.qries.com/images/banner_logo.png','c20', 'a man in orange', 'no wikipedia article found'],
        ['a21', 'https://www.qries.com/images/banner_logo.png','c21', 'a man in orange', 'no wikipedia article found'],
        ['a22', 'https://www.qries.com/images/banner_logo.png','c22', 'a man in orange', 'no wikipedia article found'],
        ['a23', 'https://www.qries.com/images/banner_logo.png','c23', 'a man in orange', 'no wikipedia article found'],
        ['a24', 'https://www.qries.com/images/banner_logo.png','c24', 'a man in orange', 'no wikipedia article found'],
        ['a25', 'https://www.qries.com/images/banner_logo.png','c25', 'a man in orange', 'no wikipedia article found'],
        ['a26', 'https://www.qries.com/images/banner_logo.png','c26', 'a man in orange', 'no wikipedia article found'],
        ['a27', 'https://www.qries.com/images/banner_logo.png','c27', 'a man in orange', 'no wikipedia article found'],
        ['a28', 'https://www.qries.com/images/banner_logo.png','c28', 'a man in orange', 'no wikipedia article found'],
        ['a29', 'https://www.qries.com/images/banner_logo.png','c29', 'a man in orange', 'no wikipedia article found'],
        ['a30', 'https://www.qries.com/images/banner_logo.png','c30', 'a man in orange', 'no wikipedia article found'],
        ['a28', 'https://www.qries.com/images/banner_logo.png','c28', 'a man in orange', 'no wikipedia article found'],
        ['a29', 'https://www.qries.com/images/banner_logo.png','c29', 'a man in orange', 'no wikipedia article found'],
        ['a30', 'https://www.qries.com/images/banner_logo.png','c30', 'a man in orange', 'no wikipedia article found'],
    ];
    let nshow: number = 10;
</script>

<TableView columns={columns} types={types} rows={rows} nshow={nshow} />
