def count_params(input_dim, hid_dim, is_plastic) -> int:
	plastic_params = hid_dim * hid_dim + 1 if is_plastic else 0
	return 3 * (input_dim * hid_dim) + 3 * (hid_dim * hid_dim) + 2 + 2 + plastic_params

print(count_params(64, 80, True))
print(count_params(64, 128, False))