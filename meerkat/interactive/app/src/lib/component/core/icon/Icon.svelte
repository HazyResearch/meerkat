<script lang="ts">
	import {
		CodeSquare,
		CodeSlash,
		Code,
		Globe2,
		Image,
		FileEarmarkImage,
		FileEarmarkFont,
		Hash,
		Magic
	} from 'svelte-bootstrap-icons';

	export let name: string = 'Globe2';
	const icons: any = {
		Globe2: Globe2,
		CodeSlash: CodeSlash,
		CodeSquare: CodeSquare,
		Code: Code,
		Image: Image,
		FileEarmarkImage: FileEarmarkImage,
		FileEarmarkFont: FileEarmarkFont,
		Hash: Hash,
		Magic: Magic
	};
</script>

<svelte:component this={icons[name]} />
