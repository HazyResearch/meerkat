<script lang="ts">
    export let data: string;
    export let classes: string = '';
    export let alt: string = 'A Meerkat audio';
</script>

<audio class={classes} src={data} alt={alt}/>