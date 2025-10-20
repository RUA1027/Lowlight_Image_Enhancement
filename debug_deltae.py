import torch
import math

# First Sharma test case
L1, a1, b1 = 50.0, 2.6772, -79.7751
L2, a2, b2 = 50.0, 0.0, -82.7485
expected_de00 = 2.0425

# Manual ΔE2000 calculation
eps = 1e-12

c1 = math.sqrt(a1**2 + b1**2)
c2 = math.sqrt(a2**2 + b2**2)
c_bar = 0.5 * (c1 + c2)

c_bar7 = c_bar**7
g = 0.5 * (1.0 - math.sqrt(c_bar7 / (c_bar7 + 25.0**7)))

a1_prime = (1.0 + g) * a1
a2_prime = (1.0 + g) * a2
c1_prime = math.sqrt(a1_prime**2 + b1**2)
c2_prime = math.sqrt(a2_prime**2 + b2**2)

h1_prime = math.atan2(b1, a1_prime)
h2_prime = math.atan2(b2, a2_prime)

print(f"c1={c1:.6f}, c2={c2:.6f}, c_bar={c_bar:.6f}")
print(f"g={g:.10f}")
print(f"a1'={a1_prime:.6f}, a2'={a2_prime:.6f}")
print(f"c1'={c1_prime:.6f}, c2'={c2_prime:.6f}")
print(f"h1'={h1_prime:.6f} rad ({math.degrees(h1_prime):.2f}°)")
print(f"h2'={h2_prime:.6f} rad ({math.degrees(h2_prime):.2f}°)")

delta_L_prime = L2 - L1
delta_C_prime = c2_prime - c1_prime

# delta_h_prime calculation
diff = h2_prime - h1_prime
if abs(diff) <= math.pi:
    delta_h_prime = diff
elif diff > math.pi:
    delta_h_prime = diff - 2.0 * math.pi
else:  # diff < -math.pi
    delta_h_prime = diff + 2.0 * math.pi

print(f"\ndelta_L'={delta_L_prime:.6f}")
print(f"delta_C'={delta_C_prime:.6f}")
print(f"delta_h'={delta_h_prime:.6f} rad ({math.degrees(delta_h_prime):.2f}°)")

delta_H_prime = 2.0 * math.sqrt(c1_prime * c2_prime) * math.sin(delta_h_prime / 2.0)
print(f"delta_H'={delta_H_prime:.6f}")

L_bar_prime = 0.5 * (L1 + L2)
C_bar_prime = 0.5 * (c1_prime + c2_prime)

# h_bar_prime calculation
h_sum = h1_prime + h2_prime
if abs(h1_prime - h2_prime) <= math.pi:
    h_bar_prime = 0.5 * h_sum
elif h_sum < 2.0 * math.pi:
    h_bar_prime = 0.5 * (h_sum + 2.0 * math.pi)
else:
    h_bar_prime = 0.5 * (h_sum - 2.0 * math.pi)

print(f"\nL_bar'={L_bar_prime:.6f}")
print(f"C_bar'={C_bar_prime:.6f}")
print(f"h_bar'={h_bar_prime:.6f} rad ({math.degrees(h_bar_prime):.2f}°)")

t = (
    1.0
    - 0.17 * math.cos(h_bar_prime - math.radians(30.0))
    + 0.24 * math.cos(2.0 * h_bar_prime)
    + 0.32 * math.cos(3.0 * h_bar_prime + math.radians(6.0))
    - 0.20 * math.cos(4.0 * h_bar_prime - math.radians(63.0))
)

delta_theta = math.radians(30.0) * math.exp(-(((math.degrees(h_bar_prime) - 275.0) / 25.0) ** 2))
r_c = 2.0 * math.sqrt((C_bar_prime**7) / (C_bar_prime**7 + 25.0**7))
r_t = -math.sin(2.0 * delta_theta) * r_c

s_L = 1.0 + (0.015 * (L_bar_prime - 50.0) ** 2) / math.sqrt(20.0 + (L_bar_prime - 50.0) ** 2)
s_C = 1.0 + 0.045 * C_bar_prime
s_H = 1.0 + 0.015 * C_bar_prime * t

print(f"\nt={t:.6f}")
print(f"delta_theta={delta_theta:.6f} rad ({math.degrees(delta_theta):.2f}°)")
print(f"R_C={r_c:.6f}")
print(f"R_T={r_t:.6f}")
print(f"S_L={s_L:.6f}, S_C={s_C:.6f}, S_H={s_H:.6f}")

kL, kC, kH = 1.0, 1.0, 1.0
l_term = delta_L_prime / (kL * s_L)
c_term = delta_C_prime / (kC * s_C)
h_term = delta_H_prime / (kH * s_H)

delta_e_squared = l_term**2 + c_term**2 + h_term**2 + r_t * c_term * h_term
delta_e = math.sqrt(max(0.0, delta_e_squared))

print(f"\nl_term={l_term:.6f}, c_term={c_term:.6f}, h_term={h_term:.6f}")
print(f"ΔE00² = {delta_e_squared:.6f}")
print(f"ΔE00 = {delta_e:.6f}")
print(f"Expected = {expected_de00:.4f}")
print(f"Error = {abs(delta_e - expected_de00):.6f}")
