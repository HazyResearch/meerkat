<script lang="ts">
    export let data: string;
    export let classes: string = '';
</script>

<audio controls={true} autoplay={false}>
    <source src={data} />
</audio>