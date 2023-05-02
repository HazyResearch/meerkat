<script>
	import LabelStudio from 'label-studio';
	// import 'label-studio/build/static/css/main.css';

	console.log('LabelStudio:', LabelStudio);

	var labelStudio = new LabelStudio('label-studio', {
		config: `
			<View>
				<Image name="img" value="$image"></Image>
				<RectangleLabels name="tag" toName="img">
					<Label value="Hello"></Label>
					<Label value="World"></Label>
				</RectangleLabels>
			</View>
		`,

		interfaces: [
			'panel',
			'update',
			'submit',
			'controls',
			'side-column',
			'annotations:menu',
			'annotations:add-new',
			'annotations:delete',
			'predictions:menu'
		],

		user: {
			pk: 1,
			firstName: 'James',
			lastName: 'Dean'
		},

		task: {
			annotations: [],
			predictions: [],
			id: 1,
			data: {
				image:
					'https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg'
			}
		},

		onLabelStudioLoad: function (LS) {
			var c = LS.annotationStore.addAnnotation({
				userGenerate: true
			});
			LS.annotationStore.selectAnnotation(c.id);
		}
	});
</script>

<div id="label-studio" />
