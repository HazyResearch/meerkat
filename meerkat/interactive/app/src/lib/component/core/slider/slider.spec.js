import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Slider from './Slider.svelte';


describe('Slider', () => {
    it('should render Slider', async () => {
        const { container, component } = render(Slider, {
            props: {
                value: 13,
                min: 0,
                max: 100,
                step: 1,
            },
        });

        // Check the slider is rendered
        expect(container).toMatchSnapshot();

        // Check the slider value is 0
        const slider = screen.getByRole('slider');
        expect(slider.value).toBe('13');

        // Check the slider value is 50
        await fireEvent.change(slider, {
            target: { value: 50 },
        });
        expect(slider.value).toBe('50');

        // Check the slider value is 100
        await fireEvent.change(slider, {
            target: { value: 100 },
        });
        expect(slider.value).toBe('100');
    });
});
