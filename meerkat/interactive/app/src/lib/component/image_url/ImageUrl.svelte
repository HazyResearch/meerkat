<script lang="ts">
    export let url: string;
    export let grayscale: boolean;

    console.log("IN_IMAGE_URL")
</script>

<div>
    hello
    <img class:grayscale src={url} alt="">
</div>
