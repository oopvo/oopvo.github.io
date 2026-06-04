---
layout: default
title: 名词百科
permalink: /glossary/
---

<div class="glossary-index">
  <header class="glossary-index-header">
    <h1>📚 名词百科</h1>
    <p class="glossary-index-desc">全站技术名词汇总，点击查看详细解释</p>
  </header>

  {% assign all_terms = site.glossary | sort: "title" %}

  {% if all_terms.size > 0 %}
  {%- comment -%} Collect unique categories {%- endcomment -%}
  {%- assign raw_cats = site.glossary | map: "categories" | join: "," | split: "," | uniq | sort -%}

  <div class="glossary-filter" id="glossary-filter">
    <button class="filter-btn active" data-cat="all">全部</button>
    {% for cat in raw_cats %}
    {%- assign trimmed = cat | strip -%}
    {%- if trimmed != "" -%}
    <button class="filter-btn" data-cat="{{ trimmed }}">{{ trimmed }}</button>
    {%- endif -%}
    {% endfor %}
  </div>

  <div class="glossary-grid" id="glossary-grid">
    {% for term in all_terms %}
    {%- assign term_cat = term.categories | first | default: "未分类" -%}
    <a href="{{ site.baseurl }}{{ term.url }}" class="glossary-card-link" data-category="{{ term_cat }}">
      <article class="glossary-card">
        <div class="glossary-card-header">
          <h2 class="glossary-card-term">{{ term.title }}</h2>
          <span class="glossary-card-cat">{{ term_cat }}</span>
        </div>
        <p class="glossary-card-excerpt">{{ term.content | strip_html | truncate: 120 }}</p>
        {% if term.related_posts %}
        <div class="glossary-card-source">📎 {{ term.related_posts | first }}</div>
        {% endif %}
      </article>
    </a>
    {% endfor %}
  </div>

  <div class="glossary-empty" id="glossary-empty" style="display:none">
    <p>该分类暂无名词</p>
  </div>

  <script>
    (function() {
      var filter = document.getElementById('glossary-filter');
      var grid = document.getElementById('glossary-grid');
      var empty = document.getElementById('glossary-empty');
      var cards = grid.querySelectorAll('.glossary-card-link');

      filter.addEventListener('click', function(e) {
        var btn = e.target.closest('.filter-btn');
        if (!btn) return;

        // Update active button
        filter.querySelectorAll('.filter-btn').forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');

        var cat = btn.dataset.cat;
        var visible = 0;

        cards.forEach(function(card) {
          if (cat === 'all' || card.dataset.category === cat) {
            card.style.display = '';
            visible++;
          } else {
            card.style.display = 'none';
          }
        });

        grid.style.display = visible > 0 ? '' : 'none';
        empty.style.display = visible > 0 ? 'none' : '';
      });
    })();
  </script>

  {% else %}
  <div class="glossary-empty">
    <p>暂无名词，敬请期待</p>
  </div>
  {% endif %}
</div>
