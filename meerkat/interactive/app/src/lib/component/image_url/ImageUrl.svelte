<script lang="ts">
    export let url: string;
    export let grayscale: boolean;
</script>

<div>
    <img class:grayscale src={url} alt="">
</div>
