<script lang="ts">

    import { CodeSquare, CodeSlash, Code, Globe2 } from 'svelte-bootstrap-icons';


	export let name: string = 'Globe2';
	const icons: any = {
		Globe2: Globe2,
        CodeSlash: CodeSlash,
        CodeSquare: CodeSquare,
        Code: Code
	};
</script>

<svelte:component this={icons[name]} />
