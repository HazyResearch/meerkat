<script>
    import Pagination from "$lib/components/pagination/Pagination.svelte";

</script>

<Pagination loaded_items={32} total_items={100} loader={async (start, end) => {}} load_widget={true} />
<Pagination total_items={100} loader={async (start, end) => {}} load_widget={false} />