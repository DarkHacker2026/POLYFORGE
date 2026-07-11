int main() {
    __asm__ volatile("134: j 134b");
    return 0;
}
