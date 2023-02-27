<script lang="ts">
	import {
		CodeSquare,
		CodeSlash,
		Code,
		Globe2,
		Image,
		FileEarmarkImage,
		FileEarmarkFont,
		FileEarmarkPdf,
		Hash,
		Magic,
		BoxFill,
		CheckSquare,
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
		Magic: Magic,
		FileEarmarkPdf: FileEarmarkPdf,
		BoxFill: BoxFill,
		CheckSquare: CheckSquare,
	};
</script>

<svelte:component this={icons[name]} />
