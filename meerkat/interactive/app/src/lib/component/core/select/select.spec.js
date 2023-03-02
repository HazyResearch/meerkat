import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Select from './Select.svelte';


describe('Select', () => {
    it('should render Select', async () => {
        const { container } = render(Select, {
            props: {
                values: ["1", "2", "3"],
                labels: ['one', 'two', 'three'],
                value: "1",
            },
        });

        expect(container).toMatchSnapshot();

        // Check the select is rendered
        const select = screen.getByRole('combobox');
        expect(select).toBeInTheDocument();

        // Check the select has the correct value
        expect(select.value).toBe("1");

        // Check the select has the correct options
        const options = screen.getAllByRole('option');
        expect(options.length).toBe(3);

        // Check the select has the correct options
        expect(options[0].value).toBe("1");
        expect(options[0].textContent).toBe("one");
        expect(options[1].value).toBe("2");
        expect(options[1].textContent).toBe("two");
        expect(options[2].value).toBe("3");
        expect(options[2].textContent).toBe("three");

        // Check the select value is 2
        await fireEvent.change(select, {
            target: { value: "2" },
        });
        expect(select.value).toBe("2");
    });
});
