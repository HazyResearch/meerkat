<script lang="ts">
	import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';

	export let data: string;
	export let isUrl: boolean = true;
	export let classes: string = '';

	// GlobalWorkerOptions.workerSrc = pdfjsWorker;
	const setWorker = async () => {
		const pdfjsWorker = await import('pdfjs-dist/build/pdf.worker.entry');
		GlobalWorkerOptions.workerSrc = pdfjsWorker;
	}

	let canvasRef;
	const wrappedGetDocument = async () => {
		return await getDocument({ data: data }).promise;
	};
	let docPromise = setWorker().then(wrappedGetDocument);

	let pagePromise = docPromise.then((doc) => doc.getPage(1));
	pagePromise.then((page) => {
		const scale = 1.5;
		const viewport = page.getViewport({ scale });

		// Prepare canvas using PDF page dimensions
		var context = canvasRef.getContext('2d');
		canvasRef.height = viewport.height;
		canvasRef.width = viewport.width;

		// Render PDF page into canvas context
		var renderContext = {
			canvasContext: context,
			viewport: viewport
		};

		page.render(renderContext);
	});
</script>


<canvas bind:this={canvasRef} class={'aspect-auto border border-slate-50 shadow-sm ' + classes}  />
